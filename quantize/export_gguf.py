#!/usr/bin/env python3
"""Export 1-bit QAT weights to GGUF Q1_0_g128 format.

Uses PrismML's llama.cpp fork as reference for the GGUF format.

Usage:
    python quantize/export_gguf.py --input quantize/checkpoints/final-1bit
"""
import argparse
import struct
import numpy as np
import torch
from pathlib import Path


# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3
GGML_TYPE_Q1_0 = 30  # Q1_0_g128 type ID in PrismML's fork

GROUP_SIZE = 128


def pack_bits_numpy(signs):
    """Pack sign bits (0/1) into bytes, 8 bits per byte, LSB first."""
    flat = signs.reshape(-1)
    # Pad to multiple of 8
    pad = (8 - len(flat) % 8) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    flat = flat.reshape(-1, 8)
    multipliers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    packed = (flat * multipliers).sum(axis=1).astype(np.uint8)
    return packed


def quantize_weight_to_q1_0(weight_tensor):
    """Quantize a weight tensor to Q1_0_g128 format.

    Returns:
        scales: FP16 array of shape (num_groups,)
        packed_signs: uint8 array of packed sign bits
    """
    weight = weight_tensor.float().numpy()
    flat = weight.reshape(-1, GROUP_SIZE)
    num_groups = flat.shape[0]

    scales = np.abs(flat).mean(axis=1).astype(np.float16)
    signs = (flat >= 0).astype(np.uint8)  # 1 for positive, 0 for negative
    packed_signs = pack_bits_numpy(signs)

    return scales, packed_signs


def write_gguf_string(f, s):
    """Write a GGUF string (length-prefixed)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to exported 1-bit weights directory")
    parser.add_argument("--output", default=None,
                        help="Output GGUF file path (default: input/model-1bit.gguf)")
    parser.add_argument("--model-name", default="qat-1bit",
                        help="Model name in GGUF metadata")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or str(input_path / "model-1bit.gguf")

    print(f"Loading weights from: {input_path}")
    weights = torch.load(input_path / "weights_q1_0_g128.pt", map_location="cpu")

    print(f"Found {len(weights)} tensors")

    # Separate quantized and full-precision tensors
    q1_tensors = {}
    fp_tensors = {}

    for name, data in weights.items():
        if isinstance(data, dict) and data.get("type") == "q1_0_g128":
            q1_tensors[name] = data
            shape = data["shape"]
            print(f"  Q1_0: {name} {list(shape)}")
        elif isinstance(data, torch.Tensor):
            fp_tensors[name] = data
            print(f"  FP16: {name} {list(data.shape)}")

    print(f"\nQuantized layers: {len(q1_tensors)}")
    print(f"Full-precision layers: {len(fp_tensors)}")

    # Calculate size
    q1_bytes = sum(
        d["scales"].numel() * 2 + d["packed_signs"].numel()
        for d in q1_tensors.values()
    )
    fp_bytes = sum(t.numel() * 2 for t in fp_tensors.values())
    total_mb = (q1_bytes + fp_bytes) / 1024 / 1024
    print(f"\nEstimated size: {total_mb:.1f} MB")
    print(f"  Quantized: {q1_bytes / 1024 / 1024:.1f} MB")
    print(f"  Full prec: {fp_bytes / 1024 / 1024:.1f} MB")

    # For now, save in a simple format that can be converted to GGUF
    # Full GGUF writing requires matching llama.cpp's exact tensor naming
    # and metadata format, which depends on the model architecture.
    #
    # TODO: Use llama.cpp's convert_hf_to_gguf.py as the base, modified
    # to write Q1_0 tensors instead of the default quantization.
    #
    # For now, save the quantized state dict that can be loaded for inference.

    simple_path = str(output_path).replace(".gguf", ".pt")
    print(f"\nSaving to: {simple_path}")
    torch.save({
        "q1_tensors": q1_tensors,
        "fp_tensors": fp_tensors,
        "format": "q1_0_g128",
        "group_size": GROUP_SIZE,
    }, simple_path)

    print("Done!")
    print()
    print("To convert to GGUF, use PrismML's llama.cpp convert script:")
    print("  cd llama.cpp")
    print("  python convert_hf_to_gguf.py --outtype q1_0 <model_dir>")


if __name__ == "__main__":
    main()

# Qwen 1-Bit

Experimental 1-bit quantization-aware training (QAT) for Qwen language models. The goal is to compress Qwen3/3.5 models to true 1-bit weights ({-d, +d} per group of 128) while preserving generation quality.

Inspired by [PrismML's Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo), which demonstrated that a 1-bit 8B model (1.15 GB) can score competitively against full-precision models (16+ GB). This repo includes PrismML's demo scripts for running their pre-built Bonsai models, plus our own QAT training pipeline.

## Status

**Work in progress.** Training pipeline is functional — loss converges, logit distributions align — but generation quality is not yet at Bonsai levels. See [Findings](#findings) below.

## Quick Start — Training

```bash
# Install dependencies
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# Local training (Qwen3.5-2B, fits on any 40GB+ GPU)
PYTHONUNBUFFERED=1 python quantize/run.py

# Cloud training (Qwen3-8B, auto-detects multi-GPU + VRAM)
PYTHONUNBUFFERED=1 python quantize/run_cloud.py --model Qwen/Qwen3-8B
```

## Quick Start — Running Bonsai (PrismML's pre-built 1-bit models)

```bash
./setup.sh
./scripts/run_llama.sh -p "What is the capital of France?"
```

See [Running Bonsai Models](#running-bonsai-models) for details.

---

## Quantization Pipeline

### Format: Q1_0_g128

Each weight is 1 sign bit. Every group of 128 weights shares one FP16 scale factor:

```
w_i = scale * (2 * bit_i - 1)    bit_i in {0, 1}
```

Effective: 1.125 bits/weight. A Qwen3-8B model compresses from 16.4 GB to ~1.15 GB.

### Training Approach

| Component | Technique |
|---|---|
| Quantization | STE (straight-through estimator) with binary {-d, +d} weights |
| Scales | Learned per group of 128 (LSQ-style, optimized via backprop) |
| Distillation | Normalized logit MSE + cosine similarity (NOT KL — it explodes at 1-bit) |
| Teacher | Frozen copy of the base model (4-bit via bitsandbytes for memory efficiency) |
| Skipped layers | Embedding + LM head kept in FP16 (critical for generation) |

### Key Files

```
quantize/
├── run.py            # Local training (single GPU, validated on 2B)
├── run_cloud.py      # Cloud training (multi-GPU, auto-detects hardware)
├── quantize_lib.py   # Core: STE quantizer, BitLinear, learned scales
├── auto_tune.py      # Hyperparameter search loop
├── evaluate.py       # Benchmark evaluation
├── export_gguf.py    # Export to Q1_0_g128 GGUF format
└── train.py          # Standalone training script
```

### Hardware Requirements

| Model | Config | Total VRAM | Example GPU |
|---|---|---|---|
| Qwen3.5-2B | BF16 teacher + student | ~10 GB | Any modern GPU |
| Qwen3-8B | 4-bit teacher + BF16 student | ~78 GB | A100 80GB or 2x48GB |
| Qwen3.5-35B | 4-bit teacher + BF16 student | ~380 GB | 8x A100 80GB |

---

## Findings

### What Works
- Loss converges: CE drops from 10+ to <0.01, cosine distance from 0.8 to 0.35
- Training moves correct answers up in logit ranking (e.g., "Paris" from rank 134k to 18k)
- Normalized MSE + cosine distillation is stable (KL divergence explodes at 1-bit over large vocabs)
- 2B model trains in ~4 hours on NVIDIA DIGITS (GB10, 128GB unified memory)

### What Doesn't Work Yet
- Generation produces empty output even after training loss converges
- The model learns next-token prediction (teacher forcing) but fails at autoregressive generation
- "Paris" reaches rank 18,016 but needs to reach rank 1 out of 151,669 tokens

### Why This Is Hard
PrismML's Bonsai uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon). Their approach is described as "mathematically grounded advances designed to preserve reasoning quality under aggressive compression." No research paper has been published. Our public approach (STE + distillation) is the best-known method but likely missing key ingredients.

### Techniques From Literature Not Yet Implemented
1. Ternary {-1, 0, +1} quantization (BitNet b1.58) — more expressive but Bonsai proves binary works
2. Confidence-aware KL divergence (BitDistiller) — adapt distillation per sample
3. Attention-level distillation — match attention maps, not just output logits
4. Hadamard rotation (QuIP#) — spread weight outliers before binarization
5. Hessian-based compensation (OBS/GPTQ) — second-order weight correction

---

## Running Bonsai Models

This repo also includes PrismML's demo scripts for running their pre-built 1-bit Bonsai models.

### Setup

```bash
./setup.sh
```

### Inference

```bash
# CLI chat
./scripts/run_llama.sh -p "Your prompt"

# Switch model size (8B default, 4B, 1.7B)
BONSAI_MODEL=4B ./scripts/run_llama.sh -p "Your prompt"

# Chat server with web UI
./scripts/start_llama_server.sh    # http://localhost:8080
```

### Building from Source

Required on aarch64 (ARM64) systems — pre-built binaries are x64 only.

```bash
./scripts/build_cuda_linux.sh      # Linux CUDA
./scripts/build_mac.sh             # macOS Metal
```

---

## References

- [PrismML Bonsai-8B Whitepaper](1-bit-bonsai-8b-whitepaper.pdf)
- [BitNet b1.58 — The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- [OneBit — Towards Extremely Low-bit LLMs](https://arxiv.org/abs/2402.11295)
- [BiLLM — Pushing the Limit of PTQ for LLMs](https://arxiv.org/abs/2402.04291)
- [BitDistiller — Sub-4-Bit LLM Self-Distillation](https://arxiv.org/abs/2402.10631)
- [QuIP# — Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)
- [Optimal Brain Surgeon (Hassibi, 1993)](https://papers.nips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)

## License

Apache 2.0 (inherited from PrismML's Bonsai-demo).

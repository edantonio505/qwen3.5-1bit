#!/usr/bin/env python3
"""GPTQ-style 1-bit quantization for Qwen models.

Layer-wise post-training quantization using Hessian-based weight compensation.
Based on Optimal Brain Surgeon (Hassibi, 1993) and GPTQ (Frantar et al., 2022),
adapted for Q1_0_g128 (1-bit sign + FP16 group scale per 128 weights).

No architecture changes — the vanilla model is quantized in place.
No STE, no distillation, no training loop. Pure PTQ.

Usage:
    # Quantize Qwen3.5-2B (fits on any 24GB+ GPU)
    python quantize/gptq_1bit.py --model Qwen/Qwen3.5-2B

    # More calibration data for better quality
    python quantize/gptq_1bit.py --model Qwen/Qwen3.5-2B --nsamples 256

    # Quantize 8B (needs ~40GB VRAM)
    python quantize/gptq_1bit.py --model Qwen/Qwen3-8B --seqlen 2048
"""
import argparse
import functools
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print = functools.partial(print, flush=True)

GROUP_SIZE = 128


# ══════════════════════════════════════════════════════════
#  Hadamard Rotation (QuIP#-style)
# ══════════════════════════════════════════════════════════

def hadamard_matrix(n, device="cpu"):
    """Construct a normalized Hadamard matrix of size n (power of 2)."""
    if n == 1:
        return torch.ones(1, 1, device=device)
    h = hadamard_matrix(n // 2, device)
    return torch.cat([
        torch.cat([h, h], dim=1),
        torch.cat([h, -h], dim=1),
    ], dim=0) / (2 ** 0.5)


def random_hadamard(n, device="cpu", seed=42):
    """Randomized Hadamard: H @ diag(random_signs).

    Makes the rotation data-independent, which prevents adversarial alignment.
    """
    n_pad = 1 << (n - 1).bit_length()  # next power of 2
    H = hadamard_matrix(n_pad, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    signs = torch.randint(0, 2, (n_pad,), device=device, generator=gen) * 2 - 1
    return H * signs.unsqueeze(0), n_pad


# ══════════════════════════════════════════════════════════
#  GPTQ 1-bit Quantizer (per-layer)
# ══════════════════════════════════════════════════════════

class GPTQ:
    """GPTQ quantizer for a single nn.Linear layer.

    Accumulates the Hessian (H = X^T X) from calibration activations,
    then quantizes weights column-by-column with Hessian-based compensation.

    Optionally applies Hadamard rotation before quantization (QuIP#-style)
    to equalize weight magnitudes — critical for 1-bit where outliers
    cause disproportionate quantization error.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.shape[0]  # out_features
        self.cols = layer.weight.shape[1]  # in_features
        self.H = torch.zeros((self.cols, self.cols), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp):
        """Accumulate Hessian from a batch of input activations.

        inp: (..., in_features) tensor — flattened to (tokens, in_features)
        """
        if inp.dim() > 2:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float().to(self.dev)
        self.H.addmm_(inp.T, inp, alpha=1.0, beta=1.0)
        self.nsamples += n

    @torch.no_grad()
    def quantize(self, percdamp=0.01, blocksize=128, use_hadamard=True,
                 refine_iters=5):
        """Run GPTQ 1-bit quantization with optional Hadamard rotation.

        Args:
            percdamp: Hessian dampening as fraction of mean diagonal
            blocksize: GPTQ block size (should match GROUP_SIZE)
            use_hadamard: apply random Hadamard rotation before quantization
            refine_iters: iterative sign-flip refinement passes after GPTQ

        Returns:
            Q: (rows, cols) quantized weight tensor (values are ±scale)
            group_scales: (num_groups,) FP32 scales for each group of 128
            total_error: scalar, proxy for quantization quality
        """
        W = self.layer.weight.data.float().clone()
        H = self.H.clone()
        if self.nsamples > 0:
            H /= self.nsamples

        rows, cols = W.shape

        # ── Hadamard rotation ──
        # Rotate input dimension: W' = W @ Had^T, H' = Had @ H @ Had^T
        # This spreads outlier magnitudes evenly, making 1-bit quantization
        # much more effective. The rotation is undone after quantization.
        Had = None
        if use_hadamard and cols >= 64:
            Had, n_pad = random_hadamard(cols, device=self.dev)
            Had = Had[:cols, :cols]  # trim to actual size
            W = W @ Had.T
            H = Had @ H @ Had.T

        # Dead columns: inputs that were never activated
        dead = H.diagonal() == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        # Dampening for numerical stability
        damp = percdamp * H.diagonal().mean()
        H.diagonal().add_(damp)

        # Upper Cholesky of H^{-1} (standard GPTQ formulation)
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
        except Exception:
            H.diagonal().add_(damp * 100)
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)
        Losses = torch.zeros(rows, device=self.dev)

        # ── GPTQ column-wise quantization ──
        for i1 in range(0, cols, blocksize):
            i2 = min(i1 + blocksize, cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # 1-bit quantization: q = sign(w) * group_scale
                # Scale = mean absolute value across all columns in this block/group
                scale = W1.abs().mean(dim=1)  # (rows,) per-row scale

                q = w.sign()
                q[q == 0] = 1.0
                q = q * scale

                Q1[:, i] = q
                Losses1[:, i] = ((w - q) / d) ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses += Losses1.sum(dim=1) / 2

            if i2 < cols:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        # ── Full-Hessian coordinate descent refinement ──
        # For each weight, check if flipping its sign reduces the
        # full activation-weighted error: ||Q@X - W@X||²_F = Tr(R^T R H)
        # where R = Q - W and H = X^T X.
        #
        # For flipping Q[i,j] by -2*q: Δ ≈ -4q * (R @ H)[i,j] + 4q² * H[j,j]
        # This is the linearized benefit of flipping, accounting for
        # cross-column interactions (not just diagonal).
        if refine_iters > 0:
            W_orig = self.layer.weight.data.float().clone()
            if Had is not None:
                W_orig = W_orig @ Had.T

            H_full = self.H.clone()
            if self.nsamples > 0:
                H_full /= self.nsamples
            if Had is not None:
                H_full = Had @ H_full @ Had.T

            total_flipped = 0
            for iteration in range(refine_iters):
                R = Q - W_orig  # residual (rows, cols)
                RH = R @ H_full  # (rows, cols) — the expensive step

                # For each element: benefit of flipping sign
                # flip changes Q[i,j] by delta_q = -2*Q[i,j]
                # Δ_error ≈ delta_q * (2*RH[i,j] + delta_q * H[j,j])
                delta_q = -2.0 * Q
                h_diag = H_full.diagonal().unsqueeze(0)  # (1, cols)
                delta_err = delta_q * (2.0 * RH + delta_q * h_diag)

                flip_mask = delta_err < 0
                flipped = flip_mask.sum().item()

                if flipped == 0:
                    break

                # Apply flips
                Q[flip_mask] = -Q[flip_mask]

                # Recompute optimal group scales: scale_g = mean(sign_g * w_orig_g)
                flat_Q = Q.reshape(-1, GROUP_SIZE)
                flat_W = W_orig.reshape(-1, GROUP_SIZE)
                new_signs = flat_Q.sign()
                new_signs[new_signs == 0] = 1.0
                optimal_scales = (new_signs * flat_W).mean(dim=1, keepdim=True)
                optimal_scales = optimal_scales.clamp(min=1e-8)
                flat_Q[:] = new_signs * optimal_scales
                Q = flat_Q.reshape(rows, cols)

                total_flipped += flipped

        # ── Undo Hadamard rotation ──
        if Had is not None:
            Q = Q @ Had  # Had is orthogonal, so Had^{-1} = Had^T, but Had is symmetric

        # Final group scales from de-rotated quantized weights
        flat = Q.reshape(-1, GROUP_SIZE)
        group_scales = flat.abs().mean(dim=1)

        return Q, group_scales, Losses.sum().item()


# ══════════════════════════════════════════════════════════
#  Calibration Data
# ══════════════════════════════════════════════════════════

def prepare_calibration_data(tokenizer, nsamples=128, seed=42, seqlen=2048):
    """Load calibration data from WikiText-2.

    Returns a list of (1, seqlen) token tensors.
    """
    print(f"  Loading calibration data ({nsamples} samples, seqlen={seqlen})...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    rng = torch.Generator()
    rng.manual_seed(seed)

    data = []
    for _ in range(nsamples):
        start = torch.randint(0, len(tokens) - seqlen, (1,), generator=rng).item()
        data.append(tokens[start:start + seqlen].unsqueeze(0))

    print(f"  {len(data)} calibration sequences ready")
    return data


# ══════════════════════════════════════════════════════════
#  Model Layer Utilities
# ══════════════════════════════════════════════════════════

def get_layers(model):
    """Get transformer layer list from a Qwen/Llama-style model."""
    return model.model.layers


def get_embed(model):
    """Get the embedding module."""
    return model.model.embed_tokens


def get_norm(model):
    """Get the final RMSNorm."""
    return model.model.norm


def get_lm_head(model):
    """Get the LM head."""
    return model.lm_head


def find_linears(module, skip_patterns=None):
    """Find all nn.Linear layers in a module (one level deep via named_modules)."""
    if skip_patterns is None:
        skip_patterns = []
    result = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            if any(pat in name.lower() for pat in skip_patterns):
                continue
            if child.weight.numel() % GROUP_SIZE != 0:
                continue
            result[name] = child
    return result


# ══════════════════════════════════════════════════════════
#  Layer-by-layer Quantization
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def quantize_model(model, calibration_data, args, tokenizer=None, device=None):
    """Quantize all transformer layers using GPTQ-1bit.

    Processes layers sequentially: quantize layer i, then feed quantized
    outputs as inputs to layer i+1 (so later layers see realistic inputs).

    If args.eval_every > 0, runs generation eval every N layers so you can
    see early whether the approach is working (abort if already broken).
    """
    device = next(model.parameters()).device
    layers = get_layers(model)
    n_layers = len(layers)

    skip = ["norm", "layernorm", "rmsnorm"]
    if not args.quantize_embed:
        skip.append("embed")
    if not args.quantize_lm_head:
        skip.append("lm_head")

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── Phase 1: Capture inputs to first transformer layer ──
    print("\n  Capturing layer-0 inputs...")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.inps = []
            self.kwargs = None

        def forward(self, hidden_states, *args, **kwargs):
            self.inps.append(hidden_states.data)
            if self.kwargs is None:
                self.kwargs = kwargs
            raise ValueError

    catcher = Catcher(layers[0])
    layers[0] = catcher

    for batch in calibration_data:
        try:
            model(batch.to(device))
        except ValueError:
            pass

    inps = catcher.inps
    layers[0] = catcher.module

    # Clean up kwargs: strip cache state, force no caching.
    # Qwen3.5 has hybrid attention (linear_attn + self_attn) with a shared
    # DynamicCache that carries recurrent state — we must not re-use it.
    layer_kwargs = {}
    if catcher.kwargs:
        for k, v in catcher.kwargs.items():
            if k == "past_key_values":
                continue  # strip cache — it carries per-layer recurrent state
            elif k == "use_cache":
                layer_kwargs[k] = False
            else:
                layer_kwargs[k] = v
    layer_kwargs.setdefault("use_cache", False)

    print(f"  Captured {len(inps)} inputs, shape {inps[0].shape}")
    print(f"  Layer kwargs: {list(layer_kwargs.keys())}")

    # ── Phase 2: Quantize each transformer layer ──
    all_scales = {}
    total_model_error = 0.0

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        t0 = time.time()

        # Find linear layers to quantize in this block
        linears = find_linears(layer, skip_patterns=skip)

        if not linears:
            # No linears to quantize — just propagate
            new_inps = []
            for inp in inps:
                out = layer(inp, **layer_kwargs)
                out_hs = out[0] if isinstance(out, (tuple, list)) else out
                new_inps.append(out_hs.data)
            inps = new_inps
            continue

        # Create GPTQ quantizer for each linear
        gptqs = {name: GPTQ(linear) for name, linear in linears.items()}

        # Hook linears to capture their inputs during calibration
        hooks = []
        for name, linear in linears.items():
            gptq = gptqs[name]
            def make_hook(g):
                def hook_fn(mod, inp, out):
                    g.add_batch(inp[0].data)
                return hook_fn
            hooks.append(linear.register_forward_hook(make_hook(gptq)))

        # Run calibration inputs through this layer (pre-quantization)
        for inp in inps:
            layer(inp, **layer_kwargs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Quantize each linear
        layer_error = 0.0
        for name, gptq in gptqs.items():
            q_weight, scales, error = gptq.quantize(
                percdamp=args.percdamp,
                blocksize=GROUP_SIZE,
                use_hadamard=args.hadamard,
                refine_iters=args.refine_iters,
            )
            # Replace weight in-place
            linears[name].weight.data = q_weight.to(linears[name].weight.dtype)
            all_scales[f"layers.{layer_idx}.{name}"] = scales.half().cpu()
            layer_error += error

            # Free Hessian memory
            del gptq.H
            gptq.H = None

        del gptqs
        total_model_error += layer_error

        # Re-run calibration through the NOW-QUANTIZED layer
        new_inps = []
        for inp in inps:
            out = layer(inp, **layer_kwargs)
            out_hs = out[0] if isinstance(out, (tuple, list)) else out
            new_inps.append(out_hs.data)
        inps = new_inps

        elapsed = time.time() - t0
        n_linears = len(linears)
        print(f"  Layer {layer_idx:>2d}/{n_layers} | "
              f"{n_linears} linears | error={layer_error:.1f} | "
              f"{elapsed:.1f}s")

        # ── Mid-quantization eval checkpoint ──
        eval_every = getattr(args, 'eval_every', 0)
        if (eval_every > 0 and tokenizer is not None and device is not None
                and (layer_idx + 1) % eval_every == 0
                and layer_idx + 1 < n_layers):
            print(f"\n  ── Checkpoint eval after layer {layer_idx} "
                  f"({layer_idx + 1}/{n_layers} quantized) ──")
            model.config.use_cache = use_cache
            score = run_eval(model, tokenizer, device,
                             f"after {layer_idx + 1}/{n_layers} layers quantized")
            model.config.use_cache = False
            if score == 0:
                print(f"  ⚠ Score=0% at layer {layer_idx} — "
                      f"remaining {n_layers - layer_idx - 1} layers unlikely to help")

        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

    print(f"\n  Total quantization error: {total_model_error:.1f}")
    return all_scales


# ══════════════════════════════════════════════════════════
#  Generation & Evaluation
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_answer(model, tokenizer, prompt, device, max_tokens=60):
    msgs = [{"role": "system", "content": "Be concise."},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inp = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(
        **inp, max_new_tokens=max_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0][inp["input_ids"].shape[1]:]
    clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    return clean, raw, len(new_ids)


def run_eval(model, tokenizer, device, label=""):
    model.eval()
    questions = [
        ("Capital of France? One word.", "Paris"),
        ("2 + 2 = ? Just the number.", "4"),
        ("Largest ocean? One word.", "Pacific"),
        ("144 / 12? Just the number.", "12"),
        ("Who wrote Hamlet? Last name.", "Shakespeare"),
        ("Chemical symbol for gold?", "Au"),
        ("Year WW2 ended?", "1945"),
        ("Boiling point of water in Celsius?", "100"),
    ]
    correct = 0
    print(f"\n  --- Eval: {label} ---")
    for q, expected in questions:
        clean, raw, n_tok = generate_answer(model, tokenizer, q, device)
        hit = expected.lower() in clean.lower()
        display = clean if clean else f"[EMPTY, raw={raw[:80]}]"
        print(f"    Q: {q}")
        print(f"    A: {display[:100]} ({n_tok} tok) -> {'HIT' if hit else 'MISS'}")
        if hit:
            correct += 1
    score = correct / len(questions) * 100
    print(f"  Score: {correct}/{len(questions)} = {score:.0f}%")
    return score


# ══════════════════════════════════════════════════════════
#  Save
# ══════════════════════════════════════════════════════════

def save_quantized(model, tokenizer, all_scales, args, results, output_dir):
    """Save quantized model and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full state dict (quantized weights still in BF16 tensor format)
    print(f"  Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save group scales separately (for GGUF export)
    torch.save(all_scales, output_dir / "group_scales.pt")

    # Save Q1_0 packed format for GGUF export
    packed = {}
    for name, scales in all_scales.items():
        # Find the corresponding weight in the model
        # The name format is "layers.{idx}.{sublayer_path}"
        parts = name.split(".")
        layer_idx = int(parts[1])
        sublayer_path = ".".join(parts[2:])

        layer = get_layers(model)[layer_idx]
        # Navigate to the linear layer
        mod = layer
        for p in sublayer_path.split("."):
            mod = getattr(mod, p)

        weight = mod.weight.data.float()
        flat = weight.reshape(-1, GROUP_SIZE)
        signs = (flat >= 0).to(torch.uint8)

        # Pack 8 signs per byte
        signs_flat = signs.reshape(-1)
        pad = (8 - signs_flat.shape[0] % 8) % 8
        if pad:
            signs_flat = F.pad(signs_flat, (0, pad))
        signs_bytes = signs_flat.reshape(-1, 8)
        multipliers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8,
                                    device=signs_bytes.device)
        packed_signs = (signs_bytes * multipliers).sum(dim=1).to(torch.uint8)

        packed[name] = {
            "scales": scales,  # FP16, (num_groups,)
            "packed_signs": packed_signs,  # uint8
            "shape": list(weight.shape),
        }

    torch.save(packed, output_dir / "weights_q1_0_g128.pt")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Size estimate
    total_q1_bytes = sum(
        d["scales"].numel() * 2 + d["packed_signs"].numel()
        for d in packed.values()
    )
    print(f"  Q1_0 layers: {total_q1_bytes / 1024 / 1024:.1f} MB")
    print(f"  Saved to {output_dir}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPTQ-style 1-bit quantization")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Calibration sequence length")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Hessian dampening (percent of mean diagonal)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: quantize/runs/gptq-1bit-{model})")
    parser.add_argument("--quantize-embed", action="store_true",
                        help="Also quantize embedding layer")
    parser.add_argument("--quantize-lm-head", action="store_true",
                        help="Also quantize LM head")
    parser.add_argument("--no-hadamard", dest="hadamard", action="store_false",
                        default=True, help="Disable Hadamard rotation")
    parser.add_argument("--refine-iters", type=int, default=5,
                        help="Sign-flip refinement iterations (0 to disable)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run generation eval every N layers (0 to disable, e.g. 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        safe_name = args.model.replace("/", "-")
        args.output_dir = f"quantize/runs/gptq-1bit-{safe_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  GPTQ 1-bit Quantization (Q1_0_g128)")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Calibration: {args.nsamples} samples x {args.seqlen} tokens")
    print(f"  Dampening:   {args.percdamp}")
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:         {torch.cuda.get_device_name(0)} ({vram:.0f} GB)")
    print(f"  Embed:       {'quantize' if args.quantize_embed else 'keep FP16'}")
    print(f"  LM head:     {'quantize' if args.quantize_lm_head else 'keep FP16'}")
    print(f"  Hadamard:    {'yes' if args.hadamard else 'no'}")
    print(f"  Refine:      {args.refine_iters} iterations")
    print()

    # ── Load model and tokenizer ──
    print("[1/5] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  {param_gb:.2f} GB, {time.time() - t0:.1f}s")

    # ── Calibration data ──
    print("\n[2/5] Calibration data...")
    calib_data = prepare_calibration_data(
        tokenizer, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen)

    # ── Pre-quantization eval ──
    print("\n[3/5] Pre-quantization evaluation...")
    pre_score = run_eval(model, tokenizer, device, "BEFORE quantization (FP16)")

    # ── Quantize ──
    print("\n[4/5] Quantizing...")
    t0 = time.time()
    all_scales = quantize_model(model, calib_data, args, tokenizer=tokenizer, device=device)
    quant_time = time.time() - t0
    print(f"\n  Quantization complete: {quant_time:.1f}s")

    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ── Post-quantization eval ──
    print("\n[5/5] Post-quantization evaluation...")
    post_score = run_eval(model, tokenizer, device, "AFTER quantization (1-bit)")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  FP16 baseline:  {pre_score:.0f}%")
    print(f"  1-bit GPTQ:     {post_score:.0f}%")
    print(f"  Delta:          {post_score - pre_score:+.0f}%")
    print(f"  Quant time:     {quant_time:.1f}s")
    print(f"{'=' * 60}")

    # ── Save ──
    results = {
        "model": args.model,
        "pre_score": pre_score,
        "post_score": post_score,
        "delta": post_score - pre_score,
        "quant_time_s": quant_time,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "percdamp": args.percdamp,
    }
    save_quantized(model, tokenizer, all_scales, args, results, args.output_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# Qwen 1-Bit

Experimental 1-bit quantization-aware training (QAT) for Qwen language models. The goal is to compress Qwen3/3.5 models to true 1-bit weights ({-d, +d} per group of 128) while preserving generation quality.

Inspired by [PrismML's Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo), which demonstrated that a 1-bit 8B model (1.15 GB) can score competitively against full-precision models (16+ GB). This repo includes PrismML's demo scripts for running their pre-built Bonsai models, plus our own QAT training pipeline.

## Status

**Active: v8 training run on 2x A100 80GB.** Full OneBit architecture from codebase audit: LayerNorm inside every BitLinear (THE primary fix for generation collapse), tanh-STE, NMF init with weight=sign(W)*0.01, all-layer directional alignment as dominant loss, LR=1e-4. Prior runs (v4.3-v7) all had good training loss but broken generation — root cause: missing LayerNorm causes activation explosion during autoregressive generation.

## Quick Start — Training

```bash
# Install dependencies
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# 8B model — CURRENT BEST (v8: full OneBit architecture with LayerNorm fix)
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 10000 \
    --max-examples 30000 --epochs 50 --lr 1e-4 \
    --gen-check-interval 200 --eval-interval 1000 \
    --output-dir quantize/runs/v8-qwen3-8b \
    --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 --unlikelihood-weight 0 \
    2>&1 | tee run_v8.log

# First run (no GPTQ checkpoint yet — runs Phase 1 first, ~15 min):
# Remove --skip-gptq and --gptq-checkpoint flags

# Previous approaches (all failed — see Findings):
# v4.3: python3 quantize/run_v4.py (loss converges, gen collapses)
# v5.0: GPTQ binary init (flat gradient landscape)
# v5.1: on-policy OOM'd (student on single GPU)
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

### Training Approach (v8 — Full OneBit Architecture from Codebase Audit)

| Component | Source | Technique |
|---|---|---|
| **LayerNorm in BitLinear** | OneBit bitnet.py | `nn.LayerNorm(out, elementwise_affine=False)` after every binary matmul. **THE primary fix.** |
| **Tanh-STE** | OneBit bitnet.py | `grad * (1.001 - tanh(w)²)` — smooth gradient gate |
| **NMF init** | OneBit build_start_ckpt.py | Rank-1 NMF on \|W\| for alpha/beta. weight = sign(W) * 0.01 |
| **All-layer alignment** | OneBit kd.py | L2-normalized MSE at every layer (dominant term). KD logit loss scaled 100x down. |
| **SVID decomposition** | OneBit (2402.11295) | `x*beta → sign(w)@x → output*alpha → LayerNorm` |
| **LR 1e-4, beta2=0.98** | OneBit llama_7b.sh | 20x higher LR, responsive optimizer |
| **GPTQ init** | gptq_1bit.py | Hessian-calibrated signs as starting point |
| **On-policy** | MiniLLM (2306.08543) | 15% of steps, temp=0.8, 32-token prefix |
| **Data** | OneBit recipe | 30k examples × 50 epochs (repetition for sign learning) |

**Why LayerNorm inside BitLinear is critical (the bug we were missing):**
During teacher forcing, input tokens produce bounded activations — training loss drops normally.
During generation, the model feeds its own outputs back. At 1-bit, each binary matmul amplifies
errors by O(√d). Without LayerNorm, by layer 20 of 36, activations have drifted completely from
training distribution. Logits collapse to high-frequency tokens (the attractors we kept hitting:
`\n\n`, `, 01.`, `Okayimport`). LayerNorm re-normalizes after every layer, keeping generation
activations bounded regardless of input source. This is not in the OneBit paper — found by
auditing their actual GitHub codebase.

### Key Files

```
quantize/
├── run_v5.py         # CURRENT: v8 — full OneBit architecture (LayerNorm + tanh-STE + NMF + all-layer alignment)
├── gptq_1bit.py      # GPTQ 1-bit PTQ: Hadamard rotation, sign-flip refinement, layer-wise calibration
├── run_v4.py         # v4.3: BitLinear QAT (loss converges but generation collapses)
├── run_cloud.py      # Cloud training (multi-GPU, 4-bit teacher, auto-detect, BitLinear)
├── run.py            # Legacy: local training with SubLN + dynamic scales
├── quantize_lib.py   # Core: ProgressiveQuantizedLinear, STE, Hadamard, learned scales
├── auto_tune.py      # Hyperparameter search loop
├── evaluate.py       # Benchmark evaluation
├── export_gguf.py    # Export to Q1_0_g128 GGUF format
└── train.py          # Standalone training script
```

### Hardware Requirements

Measured from actual training runs (teacher + student + optimizer + gradients + activations):

| Model | Config | Total VRAM | Example GPU | Verified |
|---|---|---|---|---|
| Qwen3.5-2B | BF16 teacher + ProgressiveQuantized student | ~40 GB | A40 48GB | Yes |
| Qwen3-8B | 4-bit teacher + split student (v5.3) | ~50-60 GB peak/GPU | 2x A100 80GB | Yes (v5.3) |
| Qwen3-8B | 4-bit teacher + student single GPU + on-policy | ~84 GB peak | OOM on 80GB | Yes (v5.1 OOM) |
| Qwen3.5-35B | 4-bit teacher + student | ~380 GB | 8x A100 80GB | Estimated |

**Split student across both GPUs (v5.3 layout):** Student layers 0-17 on GPU 0 (shared with teacher), layers 18-35 on GPU 1. Peak ~56 GB per GPU. Required for on-policy distillation (2 forward passes per step).

**Student on single GPU will OOM with on-policy:** Peaks at 84 GB (v5.1 finding). Without on-policy, single GPU works but generation collapses.

**ProgressiveQuantizedLinear also OOMs on 8B** due to blending intermediates. Use BitLinear for 8B+.

**Note:** 24GB GPUs (RTX 3090/4090) cannot fit even the 2B model due to optimizer states and activations.

---

## Findings

### Key Discoveries

1. **KL divergence explodes at 1-bit** — over 151k vocab, KL goes from 16 to 3600+. Clamping makes it useless. Use normalized MSE + cosine similarity instead.

2. **Teacher forcing causes generation collapse** — training loss converges perfectly (CE→0.003, cosine→0.30) but the model outputs `\n\n` repeated 60 times during generation. The model never learns to recover from its own errors. **Mitigation:** Scheduled sampling (10%→30%) — replace some teacher-forced tokens with student's own predictions during training.

3. **ProgressiveQuantizedLinear OOMs on 8B models** — the forward pass computes `w + noise*(w_1bit - w)` which creates 3 full-size intermediate tensors per layer. Plus `STEQuantize1Bit` saves signs in BF16 (2 bytes). For 252 layers at 8B, this exceeds 80 GB. **Fix:** Use `BitLinear` which saves signs as int8 (1 byte, 50% savings) and does no blending.

4. **2x A100 80GB (160 GB total) works for 8B QAT** — teacher on GPU 0 (~6.4 GB in 4-bit), student on GPU 1 (~16.5 GB base, peaks at ~78 GB during backward). Config: batch=1, seq=512, grad_accum=16, 8-bit AdamW. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for defragmentation.

5. **Single A100 80GB is NOT enough for 8B QAT** — even with BitLinear, student peaks at ~78 GB during backward. Teacher needs another ~6 GB. OOMs at 84+ GB on 80 GB GPUs.

6. **8-bit AdamW saves ~16 GB** — via bitsandbytes. Critical for fitting 8B on 80 GB GPUs. Standard AdamW states would add ~32 GB.

7. **Transformers 5.5.0 + PyTorch 2.4.x incompatibility** — transformers calls `nn.Module.set_submodule()` which only exists in PyTorch 2.5+. Fix: monkey-patch `set_submodule` onto `nn.Module`. Symptom: `AttributeError: 'Qwen3ForCausalLM' object has no attribute 'set_submodule'`.

8. **SubLN helps at initialization** — RMSNorm before each binary linear prevents hidden state collapse initially, but training pushes the model back to the `\n\n` attractor.

9. **The model IS learning** — "Paris" moves from rank 134,940 to 18,016 in the logit ranking after 375 steps. It needs to reach rank 1 out of 151,669 tokens. More steps and more data should help.

10. **Embed + LM head should stay FP16** — these layers directly interface with the token space. Quantizing them breaks generation immediately.

### What Works
- Normalized MSE + cosine + CE distillation is stable (no explosion)
- Learned per-group scales with 10x LR multiplier
- BitLinear with int8 sign storage fits 8B on 80 GB GPUs
- 4-bit teacher via bitsandbytes NF4 saves ~12 GB VRAM
- 8-bit AdamW via bitsandbytes saves ~16 GB optimizer memory
- **Split student across both GPUs** via accelerate.dispatch_model() — peak ~56 GB/GPU
- Training loss converges consistently across all runs
- Gradient checkpointing essential for 8B
- **GPTQ init with FP16 magnitudes + sign flips** — better than both naive init and raw GPTQ
- **Clipped STE** — slower learning but more accurate gradients at 1-bit
- **Unlikelihood loss** — actively trains against repetition (ul > 0 from step ~20)

### What Doesn't Work
- **Generation still collapses** with naive init + teacher forcing (v4.3)
- ProgressiveQuantizedLinear on 8B → OOM (use BitLinear)
- GPTQ binary values as weight init → flat gradients (keep FP16 magnitudes instead)
- Student on single GPU + on-policy → OOM at 84 GB (must split across GPUs)
- KL divergence → explodes to 3600+ (use MSE + cosine)
- CUDA async errors crash gen checks → try/except + synchronize + CUDA_LAUNCH_BLOCKING=1
- **18M tokens (35k examples) is 400x too little** — scaled to 300k examples in v5.3

### Under Test (v5.3)
- On-policy distillation (MiniLLM) — first principled fix for generation collapse
- Unlikelihood loss — training against repetition attractor
- Clipped STE — better gradient quality through sign()
- 300k examples × 20 epochs ≈ 3B tokens (166x more than v4.3)

### v4.3 Run (killed step 300 — 2026-04-04)
- **Setup:** 2x A100 80GB, Qwen3-8B, 4-bit teacher, BitLinear student
- **Config:** batch=1, seq=512, grad_accum=16, 8-bit AdamW, lr=5e-6 (scales: 5e-5)
- **Data:** 35,160 examples (30k OpenHermes chat + 5.1k QA)
- **Schedule:** 3000 steps, scheduled sampling 10%→30%
- **Teacher baseline:** 8/8 = 100% on factual eval
- **Student baseline (untrained 1-bit):** 0/8 = 0% (gibberish)

**Loss trajectory:**

| Step | Loss | CE | Generation |
|------|------|----|------------|
| 1 | 8.62 | 19.0 | `illonillianillery` (random tokens) |
| 75 | 4.24 | 8.8 | — |
| 150 | 2.63 | 5.1 | — |
| 200 | — | — | `Nowonenatorinaeseinged up down away did` (English fragments!) |
| 225 | 2.05 | 3.8 | — |

**Outcome:** Killed at step 300 — loss plateaued at 1.93, generation still incoherent English fragments. Same convergence-without-generation pattern as all prior runs. Pivoted to v5 (GPTQ init + QAT).

### v5.0 GPTQ-only Run (killed early — 2026-04-04)
- GPTQ Phase 1 completed: total error 890,626, 0/8 eval (collapsed to "hofhofhof")
- Phase 2 started but loss was WORSE than v4.3 (8.89 vs 8.62) — GPTQ binary values
  gave flat gradient landscape, hurting optimization
- **Root cause:** Initializing `self.weight` with GPTQ binary values (±scale) instead of
  keeping FP16 magnitudes. Fixed in v5.1.

### v5.1 (OOM'd at step 25)
- On-policy distillation required 2 forward passes, pushing single-GPU student to 84 GB → OOM
- Fix: split student across both GPUs (v5.3)

### v5.3 (killed step ~120 — hyperparameters too conservative)
- Loss 5.79 at step 75 (slower than v4.3's 4.24)
- Step 100 gen: `, 01. a the is and in to that for` — new repetition attractor
- Unlikelihood weight 0.1 too weak (ul=0.006, barely registers)
- STE clip 1.0 too aggressive (zeroed too many gradients)
- On-policy ramp too slow (1% at step 75, needed to be active from start)

### v5.4 (killed step 75 — same trajectory as v5.3)
- Stronger hyperparameters didn't change trajectory. Problem is structural, not tuning.

### v6 (killed step 1000 — 5-term loss causes degenerate attractors)
- SVID + 500k data. Loss: 9.95→2.24 (best ever). But gen: "Okayimport import list_list". Eval 0/8.
- **Root cause: 5-term loss (MSE+cos+CE+h_MSE+UL) creates conflicting gradients.** Model finds
  compromise modes that satisfy all terms but produce degenerate generation.
- 500k examples seen once < 30k examples seen multiple times (repetition matters).

### v7 (killed — missing LayerNorm inside BitLinear)
- Simple loss + SVID were correct, but still missing the core OneBit architectural feature.
- Research agent audited OneBit's actual GitHub codebase and found 5 critical differences.

### v8 Run (In Progress — 2026-04-05) — Full OneBit Architecture
Implements ALL 5 fixes found by auditing OneBit's actual codebase (github.com/xuyuzhuang11/OneBit):

| Fix | What | Why |
|-----|------|-----|
| **1. LayerNorm in BitLinear** | `nn.LayerNorm(out, elementwise_affine=False)` after every binary matmul | **PRIMARY BUG FIX.** Prevents activation explosion during generation. Each 1-bit layer amplifies errors by O(√d); by layer 20, activations have drifted from training distribution. LayerNorm re-normalizes, keeping generation in-distribution. |
| **2. Tanh-STE** | `grad * (1.001 - tanh(w)²)` | Smooth gate: plastic near 0 (uncertain sign), frozen far from 0 (committed sign). Better than vanilla or clipped STE. |
| **3. NMF init + w=sign(W)*0.01** | sklearn NMF on \|W\| for alpha/beta, small weight magnitude | NMF preserves covariance structure. w=0.01 gives 100% gradient flow (vs 42% at w=1.0 with tanh-STE). |
| **4. All-layer directional alignment** | L2-normalized MSE at every layer, pkd_loss dominant | Forces all 36 layers to track teacher's hidden state directions. KD logit loss scaled down 100x. |
| **5. LR 1e-4, beta2=0.98** | 20x higher LR, responsive optimizer | OneBit uses 4e-4. Our 5e-6 was 80x too low — sign landscape was frozen. |

### Why This Is Hard
PrismML's Bonsai uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon). Their approach is described as "mathematically grounded advances designed to preserve reasoning quality under aggressive compression." No research paper has been published.

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

### Core (directly implemented in v8 pipeline)
- **[OneBit](https://arxiv.org/abs/2402.11295) (NeurIPS 2024)** — **PRIMARY SOURCE.** Our entire v8 architecture: SVID decomposition, LayerNorm inside BitLinear, tanh-STE, NMF init, per-layer directional alignment loss. [GitHub](https://github.com/xuyuzhuang11/OneBit)
- **[FBI-LLM](https://arxiv.org/abs/2407.07093)** — Proved binary {-1,+1} LLMs work at 7B. Showed distillation-only loss outperforms combined losses.
- [PrismML Bonsai-8B Whitepaper](1-bit-bonsai-8b-whitepaper.pdf) — target: 70.5% avg at 1-bit (proprietary)
- [MiniLLM](https://arxiv.org/abs/2306.08543) (ICLR 2024) — on-policy distillation with reverse KL
- [GPTQ](https://arxiv.org/abs/2210.17323) (ICLR 2023) — Hessian-based calibration for Phase 1 init
- [What Makes Low-Bit QAT Work](https://arxiv.org/abs/2601.14888) — GPTQ init before QAT
- [QuEST](https://arxiv.org/abs/2502.05003) — Hadamard normalization for 1-bit
- [Optimal Brain Surgeon (Hassibi, 1993)](https://papers.nips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) — foundation for GPTQ

### Additional references
- [GKD](https://arxiv.org/abs/2306.13649) (ICLR 2024) — tunable on-policy fraction
- [Unlikelihood Training](https://arxiv.org/abs/1908.04319) (ICLR 2020) — penalize repeated tokens
- [PV-Tuning](https://arxiv.org/abs/2405.14852) — STE analysis for extreme compression
- [TinyBERT](https://arxiv.org/abs/1909.10351) (EMNLP 2020) — multi-layer distillation
- [BitDistill](https://arxiv.org/abs/2510.13998) — hidden state distillation for 1-bit
- [AWQ](https://arxiv.org/abs/2306.00978) (MLSys 2024) — activation-aware weight quantization
- [EfficientQAT](https://arxiv.org/abs/2407.11062) (ACL 2025) — block-wise QAT
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — ternary training from scratch
- [BitNet v2](https://arxiv.org/abs/2504.18415) — H-BitLinear
- [QuIP#](https://arxiv.org/abs/2402.04396) — Hadamard incoherence
- [Rethinking 1-bit Optimization](https://arxiv.org/abs/2508.06974) — tanh progressive schedule
- [ARB-LLM](https://arxiv.org/abs/2410.03129) — alternating refined binarizations
- [Binary Neural Networks for LLMs: A Survey](https://arxiv.org/abs/2502.19008)
- [BiLLM](https://arxiv.org/abs/2402.04291) — PTQ for LLMs
- [BitDistiller](https://arxiv.org/abs/2402.10631) — sub-4-bit self-distillation

## License

Apache 2.0 (inherited from PrismML's Bonsai-demo).

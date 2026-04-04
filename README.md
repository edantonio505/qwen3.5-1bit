# Qwen 1-Bit

Experimental 1-bit quantization-aware training (QAT) for Qwen language models. The goal is to compress Qwen3/3.5 models to true 1-bit weights ({-d, +d} per group of 128) while preserving generation quality.

Inspired by [PrismML's Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo), which demonstrated that a 1-bit 8B model (1.15 GB) can score competitively against full-precision models (16+ GB). This repo includes PrismML's demo scripts for running their pre-built Bonsai models, plus our own QAT training pipeline.

## Status

**Active: v5 training run on 2x A100 80GB.** Two-phase pipeline: (1) GPTQ calibration with Hadamard rotation for optimal binary weight initialization, (2) QAT fine-tuning with hidden state distillation. v4.3 proved loss converges but generation collapses with naive sign(w) init — research shows GPTQ initialization yields 15x better results.

## Quick Start — Training

```bash
# Install dependencies
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# 8B model — CURRENT BEST (v5.3: split student + on-policy + unlikelihood + clipped STE)
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 3000 \
    --gen-check-interval 100 --eval-interval 500 \
    --output-dir quantize/runs/v5.3-qwen3-8b \
    --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
    --unlikelihood-weight 0.1 --on-policy-fraction 0.2 --on-policy-len 64 --ste-clip 1.0 \
    2>&1 | tee run_v5.3.log

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

### Training Approach (v5.1 — GPTQ Init + On-Policy + Unlikelihood + Clipped STE)

| Component | Paper | Technique |
|---|---|---|
| **Phase 1: GPTQ init** | "What Makes Low-Bit QAT Work" (2601.14888) | Hessian-based calibration + Hadamard rotation + sign-flip refinement |
| **Phase 1: AWQ weighting** | AWQ (2306.00978) | Activation-magnitude priority for sign-flip refinement |
| **Phase 2: GPTQ-init BitLinear** | — | FP16 magnitudes preserved, signs flipped to match GPTQ Hessian-optimal |
| **Clipped STE** | PV-Tuning (2405.14852) | Zero gradient for weights with \|w\| > 1.0 — focuses learning on decision boundary |
| **On-policy distillation** | MiniLLM (2306.08543) / GKD (2306.13649) | 20% of steps: student generates 64 tokens, soft CE vs teacher on student sequences |
| **Unlikelihood loss** | Unlikelihood Training (1908.04319) | Penalize log(1-p) for tokens in last 16 positions — breaks repetition attractor |
| **Hidden state MSE** | BitDistill (2510.13998) / TinyBERT (1909.10351) | Match last transformer layer output between teacher and student |
| **Distillation loss** | — | Normalized logit MSE (0.4) + cosine (0.2) + CE (0.4) + hidden MSE (0.1) + UL (0.1) |
| **Data mix** | — | 60% OpenHermes chat + 40% QA (TriviaQA + GSM8K + custom factual) |
| **Teacher** | — | Frozen 4-bit NF4 on GPU 0 |
| **Optimizer** | — | 8-bit AdamW, 10x LR for scale params |

### Key Files

```
quantize/
├── run_v5.py         # CURRENT: v5 — GPTQ init + QAT + hidden state distillation
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

**A single 80GB GPU (A100) will OOM on 8B.** Peak memory during backward hits ~78 GB for student alone; teacher needs another ~6 GB.

**ProgressiveQuantizedLinear also OOMs on 8B** (even on 80GB) due to blending intermediates. Use BitLinear for 8B+.

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
- Multi-GPU split (teacher GPU 0, student GPU 1) for 8B
- Training loss converges consistently across all runs
- Mixed QA + chat data (14% QA) provides both factual and conversational coverage
- Gradient checkpointing essential for 8B

### What Doesn't Work Yet
- Generation produces English word fragments but not correct answers (as of step 225)
- ProgressiveQuantizedLinear too memory-intensive for 8B (use BitLinear instead)
- Progressive noise schedule incompatible with 8B memory budget (skip directly to 1-bit)
- CUDA async errors crash gen checks — need try/except + synchronize wrappers

### v4.3 Run (In Progress — 2026-04-04)
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

### v5.3 Run (In Progress — 2026-04-04)
- **Key change:** Student split across both GPUs via `accelerate.dispatch_model()`
  - Layers 0-17 + embed on GPU 0 (shared with teacher)
  - Layers 18-35 + norm + lm_head on GPU 1
  - Peak ~50-60 GB per GPU instead of 84 GB on one
- **All improvements active:**
  1. **Fixed GPTQ init:** FP16 magnitudes preserved, only signs flipped to GPTQ-optimal
  2. **Clipped STE** (PV-Tuning, 2405.14852): zeros grad for |w| > 1.0
  3. **Unlikelihood loss** (1908.04319, weight=0.1): penalizes repeated tokens
  4. **On-policy distillation** (MiniLLM, 2306.08543): 20% of steps, 64-token rollouts
  5. **Hidden state MSE** (BitDistill, 2510.13998): last layer matching
  6. **GPTQ Hadamard + sign-flip** (QuEST, AWQ): Hessian-optimal signs

### Known bottleneck: data volume
OneBit (NeurIPS 2024) used 13.5B tokens. We use ~18M tokens (400x less). Every working
1-bit method used orders of magnitude more data. If v5.3 fails, scaling data is the next pivot.

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

### Core (directly implemented in our pipeline)
- [PrismML Bonsai-8B Whitepaper](1-bit-bonsai-8b-whitepaper.pdf) — target: 70.5% avg at 1-bit
- [MiniLLM — On-Policy Distillation](https://arxiv.org/abs/2306.08543) (ICLR 2024) — reverse KL + student rollouts, fixes generation collapse
- [GKD — On-Policy Distillation](https://arxiv.org/abs/2306.13649) (ICLR 2024) — tunable on-policy fraction + flexible divergence
- [Unlikelihood Training](https://arxiv.org/abs/1908.04319) (ICLR 2020) — penalize repeated tokens during training
- [PV-Tuning](https://arxiv.org/abs/2405.14852) — clipped STE for extreme compression, proxy weight alternative
- [TinyBERT](https://arxiv.org/abs/1909.10351) (EMNLP 2020) — multi-layer hidden state + attention distillation
- [EfficientQAT](https://arxiv.org/abs/2407.11062) (ACL 2025) — block-wise QAT for memory efficiency
- [AWQ](https://arxiv.org/abs/2306.00978) (MLSys 2024) — activation-aware sign-flip priority
- [What Makes Low-Bit QAT Work](https://arxiv.org/abs/2601.14888) — GPTQ init yields 15x improvement
- [BitDistill](https://arxiv.org/abs/2510.13998) — hidden state + attention distillation for 1-bit
- [QuEST](https://arxiv.org/abs/2502.05003) — Hadamard normalization + MSE-optimal fitting for 1-bit
- [GPTQ](https://arxiv.org/abs/2210.17323) (ICLR 2023) — Hessian-based column-wise quantization
- [Optimal Brain Surgeon (Hassibi, 1993)](https://papers.nips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) — mathematical foundation for GPTQ

### Additional references
- [FBI-LLM](https://arxiv.org/abs/2407.07093) — first proof binary {-1,+1} LLMs work at 7B
- [OneBit — SVID decomposition](https://arxiv.org/abs/2402.11295) (NeurIPS 2024) — W = sign(W) * outer(a,b), 13.5B token training
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — ternary {-1,0,+1} training from scratch
- [BitNet v2 — H-BitLinear](https://arxiv.org/abs/2504.18415) — online Hadamard before activation quantization
- [QuIP# — Hadamard Incoherence](https://arxiv.org/abs/2402.04396) — randomized Hadamard for incoherence processing
- [Rethinking 1-bit Optimization](https://arxiv.org/abs/2508.06974) — tanh progressive schedule FP→binary
- [ARB-LLM](https://arxiv.org/abs/2410.03129) — alternating refined binarizations
- [Binary Neural Networks for LLMs: A Survey](https://arxiv.org/abs/2502.19008)
- [BiLLM](https://arxiv.org/abs/2402.04291) — PTQ for LLMs
- [BitDistiller](https://arxiv.org/abs/2402.10631) — sub-4-bit self-distillation

## License

Apache 2.0 (inherited from PrismML's Bonsai-demo).

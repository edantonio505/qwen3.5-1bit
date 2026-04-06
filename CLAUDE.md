# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project has two parts:
1. **Bonsai Demo** — local inference runner for PrismML's 1-bit quantized models
2. **1-bit Quantization Research** — building our own 1-bit quantization pipeline to reproduce/understand PrismML's Bonsai results

### Inference (Bonsai Demo)

Local inference runner for PrismML's 1-bit quantized language models (8B, 4B, 1.7B). Two backends:
- **llama.cpp** (GGUF format) — C/C++, runs on Mac (Metal), Linux/Windows (CUDA), and CPU
- **MLX** (MLX format) — Python, Apple Silicon only

Both backends use PrismML forks (not upstream) because the required 1-bit inference kernels aren't in mainline yet:
- llama.cpp: `PrismML-Eng/llama.cpp` (branch `prism`)
- MLX: `PrismML-Eng/mlx` (branch `prism`)

## Setup and Running

```bash
# Full setup (installs uv, creates venv, downloads models + binaries)
./setup.sh

# Run inference
./scripts/run_llama.sh -p "Your prompt"                    # llama.cpp (auto-detects platform)
source .venv/bin/activate && ./scripts/run_mlx.sh -p "..."  # MLX (macOS only)

# Servers
./scripts/start_llama_server.sh       # OpenAI-compatible API + chat UI on :8080
./scripts/start_mlx_server.sh         # MLX server on :8081
./scripts/start_openwebui.sh          # Open WebUI on :9090 (auto-starts backends)

# Switch model size (default: 8B)
BONSAI_MODEL=4B ./scripts/run_llama.sh -p "..."
BONSAI_MODEL=1.7B ./scripts/download_models.sh  # download first if not yet fetched

# Build from source (instead of pre-built binaries)
./scripts/build_mac.sh               # macOS Metal → bin/mac/
./scripts/build_cuda_linux.sh        # Linux CUDA → bin/cuda/
```

## Architecture

**All shell scripts source `scripts/common.sh`** which provides:
- `BONSAI_MODEL` env var handling (valid: 8B, 4B, 1.7B)
- Model path resolution (`models/gguf/{size}/` for GGUF, `models/Bonsai-{size}-mlx/` for MLX)
- Assertion helpers (`assert_valid_model`, `assert_gguf_downloaded`, `assert_mlx_downloaded`)
- Smart context size calculation based on system RAM (fallback when `-c 0` auto-fit unsupported)
- Colored logging (`info`, `warn`, `err`, `step`)
- `download()` wrapper supporting both curl and wget

**Binary layout:** `bin/mac/` for macOS Metal, `bin/cuda/` for Linux/Windows CUDA builds. Pre-built binaries come from GitHub releases; source builds output to the same directories.

**Python dependencies** are managed by **uv** (>= 0.7.0) with `pyproject.toml`. The venv lives at `.venv/`. MLX has additional heavy dependencies (torch, transformers, mlx-lm) installed during macOS setup.

## Key Details

- `BONSAI_MODEL` env var controls model size across all scripts (default: `8B`)
- Context window goes up to 65,536 tokens; `-c 0` auto-fits KV cache to available memory
- `setup.sh` is idempotent — safe to re-run, skips completed steps
- Models downloaded to `models/` and binaries to `bin/` are gitignored
- The `mlx_generate.py` script handles MLX inference with streaming output and statistics

## Quantization Research (`quantize/`)

Building a 1-bit quantization pipeline targeting Q1_0_g128 format (1 sign bit + FP16 scale per 128 weights = 1.125 bits/weight). PrismML's Bonsai achieves 70.5% avg benchmark at 1-bit vs 79.3% FP16 using proprietary Caltech IP.

### Running on Cloud GPU (RunPod / 2x A100)

```bash
git clone https://github.com/edantonio505/qwen3.5-1bit.git
cd qwen3.5-1bit
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# CURRENT best command for 8B (v8: full OneBit architecture):
pip install scikit-learn  # needed for NMF init
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 10000 \
    --max-examples 30000 --epochs 50 --lr 1e-4 \
    --gen-check-interval 200 --eval-interval 1000 \
    --output-dir quantize/runs/v8-qwen3-8b \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 --unlikelihood-weight 0 \
    2>&1 | tee run_v8.log

# If GPTQ checkpoint exists, add: --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint
# Previous versions (all failed at generation — see run history below)
```

If OOM, reduce batch/seq: `--batch-size 1 --seq-len 512`

### Quantization Format: Q1_0_g128

Every weight is binary: `w_i = scale_g * (2*bit_i - 1)` where `bit_i ∈ {0,1}` and `scale_g` is a shared FP16 scale per group of 128 weights. This is the format PrismML uses for Bonsai models.

### What We Learned (Critical Knowledge)

**KL divergence is WRONG for 1-bit distillation:**
- Over 151k vocab, KL explodes from 16 to 3600+. Clamping makes it zero-gradient.
- Use normalized MSE + cosine similarity instead. Stable, no explosion.

**Teacher forcing causes generation collapse:**
- Training loss converges perfectly (CE→0.003, cosine→0.30)
- But model outputs `\n\n` repeated 60 times during generation
- The model never sees its own errors during training
- **Mitigation:** Scheduled sampling (10%→30%) mixes student's own predictions into training inputs

**ProgressiveQuantizedLinear OOMs on 8B — use BitLinear instead:**
- `ProgressiveQuantizedLinear` blends full-precision and 1-bit weights: `w + noise*(w_1bit - w)`
- This creates 3 intermediate tensors per layer during forward pass
- `STEQuantize1Bit` in quantize_lib.py saves signs in BF16 (2 bytes each)
- For 252 layers of 8B, the intermediates + saved tensors exceed 80 GB on a single GPU
- **Fix:** `BitLinear` (in run_v4.py and run_cloud.py) saves signs as int8 (1 byte = 50% savings), does no blending, pure 1-bit from start
- Peak memory with BitLinear: **78 GB** vs 84+ GB (OOM) with ProgressiveQuantizedLinear

**2x A100 80GB split-student layout (v5.3, confirmed working):**
- Teacher (4-bit NF4) on GPU 0: ~6.4 GB
- Student layers 0-17 + embed on GPU 0: ~8 GB → peaks ~42 GB with optimizer/grads
- Student layers 18-35 + norm + lm_head on GPU 1: ~8 GB → peaks ~39 GB with optimizer/grads
- Split via `accelerate.dispatch_model()` with manual device_map
- Total peak ~56 GB per GPU — fits on-policy distillation (2 forward passes per step)
- Config: batch=1, seq=512, grad_accum=16, 8-bit AdamW
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `CUDA_LAUNCH_BLOCKING=1`

**Student on single GPU will OOM with on-policy distillation:**
- Student alone peaks at ~78 GB during backward (v4.3 config)
- On-policy adds a second forward pass → 84 GB → OOM on 80 GB GPU
- **Must split student across both GPUs** for on-policy to work
- Without on-policy, single GPU works but generation collapses (v4.3 result)

**GPTQ binary init hurts optimization (v5.0 finding):**
- Initializing BitLinear.weight with GPTQ quantized values (±scale) gives flat gradient landscape
- v5.0 starting loss was WORSE than v4.3 (8.89 vs 8.62)
- **Fix (v5.1+):** Keep original FP16 weight magnitudes, only flip signs to match GPTQ Hessian-optimal
- `gptq_signs = gptq_weight.sign(); flip where gptq_signs != orig_signs; negate weight at those positions`
- Smooth FP16 landscape for optimizer + Hessian-optimal sign decisions

**Data volume is a critical bottleneck:**
- Every working 1-bit method used 400-70,000x more data than our early runs (18M tokens)
- OneBit (NeurIPS 2024): 13.5B tokens (132k examples × 2048 seq × 50 epochs)
- FBI-LLM: 108B tokens on 16-32 A100s
- v5.3 raised to 300k examples (from 30k) × 512 seq × 20 epochs ≈ 3B tokens
- OpenHermes has 1M examples — we only use 300k. Can scale further if needed.
- SlimPajama (627B tokens) and FineWeb-Edu (1.3T tokens) available on HuggingFace for future scaling

**Memory optimizations (all applied in run_v4.py and run_cloud.py):**
- Student loaded on CPU first, quantized, then moved to GPU
- Scales computed in BF16 (no FP32 temp copies)
- STE1Bit saves signs as int8 (not BF16) — 50% saved tensor reduction
- Teacher logits detached and deleted before backward
- `del s_out` after extracting logits (frees KV cache, hidden states)
- `del s_logits, t_logits` before backward (only loss graph needed)
- `zero_grad(set_to_none=True)` to free gradient memory
- 4-bit teacher via bitsandbytes NF4
- 8-bit AdamW via bitsandbytes (saves ~16 GB optimizer memory)
- `gradient_checkpointing_enable()` on student
- Aborts after 3 consecutive OOMs with actionable message

**Transformers 5.5.0 compatibility:**
- Requires `nn.Module.set_submodule()` which is only in PyTorch 2.5+
- If using PyTorch 2.4.x, run_v4.py includes a monkey-patch for this
- Symptom: `AttributeError: 'Qwen3ForCausalLM' object has no attribute 'set_submodule'`

**CUDA illegal memory access during generation checks:**
- Async CUDA errors from training surface at `torch.cuda.empty_cache()` / `synchronize()`
- Happens at gen check boundaries (step 200, 400, etc.) when switching from train→eval→generate
- **Fix:** Wrap gen checks and evals in try/except, add `torch.cuda.synchronize()` before gen
- `CUDA_LAUNCH_BLOCKING=1` env var makes kernels synchronous (pinpoints errors, ~30% slower)
- After fix, training continues even if gen check fails

**Complete run history:**

v4.3 (killed step 300) — naive sign(w) init, single-GPU student:
- Loss: 8.6→4.2→2.6→2.0→1.9 (plateaued). Gen: English fragments, never answers.
- Root cause: naive initialization + teacher forcing exposure bias.

v5.0 (killed step 3) — GPTQ binary values as weights:
- Loss 8.89 (WORSE than v4.3). GPTQ binary values gave flat gradient landscape.
- Root cause: should keep FP16 magnitudes, only flip signs.

v5.1 (OOM step 25) — on-policy distillation on single GPU:
- On-policy rollout = 2 forward passes → 84 GB → OOM on 80 GB GPU.
- Root cause: student on single GPU can't fit on-policy.

v5.3 (killed step ~120) — split student + all improvements + 300k data:
- Loss: 8.94→5.79 at step 75 (declining but slower than v4.3's 4.24)
- Step 100 gen: `, 01. a the is and in to that for` — new repetition attractor
- Unlikelihood too weak (ul=0.006 at weight 0.1, barely registers on loss 5.79)
- On-policy too slow to ramp (op=0.01 at step 75, only 1% of steps)
- STE clip=1.0 too aggressive — zeroed too many gradients, slowed learning
- GPU stable at 42/56 GB — infrastructure works, hyperparameters were wrong

v5.4 (killed step 75) — stronger hyperparameters, same trajectory:
- Loss 5.86 at step 75 — same as v5.3. Tuning hyperparameters didn't help.
- Confirmed: the problem is structural, not hyperparameters.

v6 (killed step 1000) — SVID + 500k data + 10k steps:
- SVID decomposition: each weight gets unique scale (a_i × b_j). Lower MSE/cos from step 1.
- Loss: 9.95→4.89→2.87→2.24 (steepest drop of any run, still declining at step 1000)
- BUT gen at step 1000: "Okayimport import list_list" — new degenerate attractor. Eval 0/8.
- **Root cause: 5-term loss (MSE+cos+CE+h_MSE+UL) creates conflicting gradients.**
  Model finds compromise that satisfies all terms but produces degenerate generation.
- FBI-LLM proved distillation-only loss outperforms combined losses.

v7 (killed — missing OneBit's core architecture):
- Simple loss + SVID worked for training, but still 0/8 gen because we were missing
  the critical architectural feature: LayerNorm inside every BitLinear.

v8 (running, step 250) — Full OneBit architecture from their actual codebase:
- **FIX 1 (PRIMARY BUG):** LayerNorm(elementwise_affine=False) inside every SVIDBitLinear
  → Prevents activation magnitude explosion during autoregressive generation
  → Without this, each 1-bit layer amplifies errors by O(√d), model diverges by layer 20
  → THIS is why every prior run had good loss but broken generation
- **FIX 2:** Tanh-STE: `grad * (1.001 - tanh(w)²)` — smooth gate, plastic near 0, frozen far
- **FIX 3:** NMF init for alpha/beta + weight = sign(W) * 0.01 (max gradient flow at start)
- **FIX 4:** All-layer normalized directional alignment (L2-norm MSE at every layer, dominant term)
  → pkd_loss is the main signal, KD logit loss scaled down 100x
- **FIX 5:** LR 1e-4 (was 5e-6, 20x increase), adam_beta2=0.98 (more responsive to sign flips)
- **Early results (step 250):** pkd_loss dropped 65.6→34.8 (47% reduction in all-layer directional error)
  KD loss halved 4296→2220. Still in warmup (LR at 5e-5, target 1e-4).
  Gen at step 200: function words (`, the to a and 0 in for`). Waiting for step 400+ gen check.

**v5 approach: GPTQ init + on-policy distillation + unlikelihood + clipped STE:**

v8 implements OneBit's actual architecture (from GitHub codebase audit, not just the paper):

| Fix | Source File | arXiv | What It Does |
|-----|-------------|-------|-------------|
| **LayerNorm in BitLinear** | OneBit bitnet.py | 2402.11295 | **PRIMARY FIX.** Re-normalizes activations every layer → prevents generation collapse |
| **Tanh-STE** | OneBit bitnet.py | 2402.11295 | Smooth gradient gate: plastic near 0, frozen far from 0 |
| **NMF init + w=sign(W)*0.01** | OneBit build_start_ckpt.py | 2402.11295 | Non-negative factorization + max gradient flow at start |
| **All-layer directional alignment** | OneBit kd.py | 2402.11295 | L2-normalized MSE every layer (dominant). KD logit loss 100x down |
| **LR 1e-4, beta2=0.98** | OneBit llama_7b.sh | 2402.11295 | 20x higher LR, responsive optimizer |
| On-policy distillation | MiniLLM | 2306.08543 | Student generates during training (15% of steps) |
| GPTQ Phase 1 | GPTQ + QuEST | 2210.17323 / 2502.05003 | Hessian calibration + Hadamard for initial signs |
| Binary LLM proof | FBI-LLM | 2407.07093 | Proved {-1,+1} works at 7B + distillation-only loss is best |

**CRITICAL INSIGHT: LayerNorm inside BitLinear prevents generation collapse (v8 finding):**
OneBit's actual codebase (bitnet.py) has `nn.LayerNorm(out_features, elementwise_affine=False)`
inside EVERY BitLinear layer, applied AFTER the binary matmul and scaling. This is THE primary
fix for generation collapse. Without it, each 1-bit layer amplifies activation errors by O(√d).
During teacher forcing, input activations are bounded (correct tokens). During generation, the
model feeds its own outputs back, and small errors compound through 36 layers exponentially.
By layer 20, activations have drifted far from training distribution → logit collapse to
high-frequency tokens. LayerNorm re-normalizes after every layer, keeping activations bounded
regardless of whether input came from teacher or student's own generation.

**CRITICAL INSIGHT: Tanh-STE > vanilla STE > clipped STE (OneBit codebase):**
OneBit uses `grad * (1.001 - tanh(w)²)` — a smooth gate. Weights near zero (uncertain sign)
get full gradient. Weights far from zero (committed sign) get suppressed gradient. This is
better than vanilla STE (all weights equal) or our clipped STE (hard cutoff at |w|>1.0).

**CRITICAL INSIGHT: Weight = sign(W) * 0.01 (not sign(W) * 1.0):**
From OneBit's build_start_ckpt.py. Small magnitude means tanh-STE gives 100% gradient flow
at start → maximum exploration of sign landscape. Weights at ±1.0 only get 42% gradient.
Our GPTQ init set weights to FP16 magnitudes (large) → sign landscape was frozen from step 1.

**CRITICAL INSIGHT: NMF for alpha/beta initialization (not RMS):**
OneBit uses rank-1 NMF on |W| to initialize value vectors. NMF enforces non-negativity
(matching magnitude semantics) and captures covariance structure. RMS discards this.

**CRITICAL INSIGHT: All-layer directional alignment is the dominant loss:**
OneBit's actual loss (from kd.py): pkd_loss (per-layer L2-normalized MSE) is the DOMINANT term.
KD logit loss is scaled down 100x (kd_loss_scale=0.01). Our previous runs had logit matching
as dominant and no intermediate layer alignment → layers 1-35 could develop arbitrary internal
representations that collapsed during generation.

**CRITICAL INSIGHT: LR was 80x too low:**
OneBit uses 4e-4. We used 5e-6. At 5e-6, weight magnitudes barely move from initialization,
meaning the sign landscape is frozen at whatever GPTQ gave us. v8 uses 1e-4.

**Data:** OneBit uses 132k teacher-generated synthetic examples, 50 epochs. We use 30k
OpenHermes × 50 epochs. Future improvement: generate synthetic data from teacher.

**PrismML Bonsai (reference target):**
- Built from Qwen3-8B (standard dense transformer, NOT Qwen3.5 hybrid)
- True binary {-d, +d}, Q1_0_g128 applied to ALL layers including embed + LM head
- Uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon)
- 1-bit Bonsai 8B: 1.15 GB, 70.5 avg benchmark, 5-8x faster inference
- Whitepaper: `1-bit-bonsai-8b-whitepaper.pdf`

### Quantization Files

```
quantize/
├── run_v5.py          # CURRENT: v8 — full OneBit architecture (LayerNorm + tanh-STE + NMF + all-layer alignment)
├── run_v4.py          # v4.3 QAT with BitLinear — loss converges but gen collapses
├── gptq_1bit.py       # GPTQ 1-bit PTQ with Hadamard rotation + sign-flip refinement
├── run_cloud.py       # Cloud QAT with BitLinear + 4-bit teacher, auto-detects GPUs/VRAM/arch
├── run.py             # Local QAT with SubLN + dynamic scales (for DIGITS)
├── quantize_lib.py    # Shared library: ProgressiveQuantizedLinear, Hadamard, learned scales, STE
├── auto_tune.py       # Hyperparameter search loop (8 configs)
├── evaluate.py        # Benchmark evaluation
├── export_gguf.py     # Export to Q1_0_g128 GGUF format
└── train.py           # Standalone training script
```

### Training Architecture (run_v5.py)

**Phase 1 — GPTQ Calibration (runs once, ~10 min):**
- Loads FP16 model, runs 128 WikiText-2 calibration samples
- Layer-by-layer Hessian-based quantization (column-wise GPTQ, arXiv 2210.17323)
- Hadamard rotation before binarization (QuIP#/QuEST, spreads outlier energy)
- 5 iterations of sign-flip refinement (coordinate descent on full Hessian)
- Activation-weighted priority for sign flips (AWQ, arXiv 2306.00978)
- Saves calibrated checkpoint with optimal binary weights + group scales

**Phase 2 — QAT Fine-tuning (v8 — current best, multi-GPU):**
- **Quantizer:** SVIDBitLinear with LayerNorm inside every layer (OneBit full architecture):
  `x_scaled = x * beta → signs = TanhSTE(w) → output = linear(x_scaled, signs) * alpha → LayerNorm(output)`
- **LayerNorm:** `nn.LayerNorm(out_features, elementwise_affine=False)` — THE critical fix
- **STE:** Tanh-STE: `grad * (1.001 - tanh(w)²)` — smooth gate, not vanilla pass-through
- **Init:** weight = sign(W) * 0.01 (max gradient flow), alpha/beta from NMF on |W|
- **Loss:** `--simple-loss` — KD logit loss (0.01 weight) + per-layer directional alignment (1.0 weight)
  - All-layer L2-normalized MSE is DOMINANT (from OneBit's kd.py)
  - KD logit loss scaled down 100x
- **On-policy:** 15% of steps, temp=0.8 from 32-token prefix
- **Data:** 30k examples × 50 epochs
- **LR:** 1e-4 (20x higher than v7), beta2=0.98 (responsive to sign flips)
- **Teacher:** Frozen 4-bit NF4 on GPU 0
- **Student:** Split across both GPUs via accelerate.dispatch_model()
- **Optimizer:** 8-bit AdamW, 10x LR for SVID alpha/beta

### GPU Requirements

| Model | Config | Total VRAM | Example GPU |
|-------|--------|-----------|-------------|
| Qwen3.5-2B | BF16 teacher + ProgressiveQuantized student | ~40 GB | A40 48GB |
| Qwen3-8B | 4-bit teacher + split student (v5.3) | ~50-60 GB peak/GPU | 2x A100 80GB |
| Qwen3-8B | 4-bit teacher + student single GPU (v5.1, OOM) | ~84 GB peak | OOM on 80GB |
| Qwen3.5-35B | 4-bit teacher + student | ~380 GB | 8x A100 80GB |

**v5.3 GPU layout (recommended):** Split student across both GPUs using `accelerate.dispatch_model()`.
Teacher (6.4 GB) shares GPU 0 with first half of student layers. On-policy distillation fits because
each GPU only holds half the student's activations during forward pass.

**Student on single GPU will OOM with on-policy distillation** — two forward passes (rollout + main) exceed 80 GB.

### Operating Guide (for Claude Code sessions)

**Goal:** Quantize Qwen3-8B to true 1-bit ({-1,+1}). Target: PrismML Bonsai's 70.5% avg benchmark.
True binary only — NEVER ternary {-1,0,+1}.

**When resuming on a new server or after a crash:**
1. Check GPU setup: `nvidia-smi` — need 2x 80GB+ GPUs
2. Check deps: `python3 -c "import torch, transformers, bitsandbytes, accelerate; print('OK')"`
3. Install extras: `pip install scikit-learn tensorboard`
4. Check what checkpoints exist:
   ```bash
   ls quantize/runs/v8-qwen3-8b/checkpoint-*/training_state.pt 2>/dev/null  # periodic checkpoints
   ls quantize/runs/v8-qwen3-8b/best/model.safetensors 2>/dev/null          # best eval checkpoint
   ls quantize/runs/v5-qwen3-8b/gptq_checkpoint/group_scales.pt 2>/dev/null  # GPTQ init
   ```

5. **If periodic checkpoint exists (checkpoint-XXXX/training_state.pt):**
   Resume from exact step with optimizer state:
   ```bash
   PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python3 quantize/run_v5.py \
       --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 10000 \
       --max-examples 30000 --epochs 50 --lr 1e-4 \
       --gen-check-interval 200 --eval-interval 1000 \
       --output-dir quantize/runs/v8-qwen3-8b \
       --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
       --use-svid --simple-loss \
       --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 --unlikelihood-weight 0 \
       --resume-from quantize/runs/v8-qwen3-8b/checkpoint-XXXX \
       2>&1 | tee run_v8_resumed.log
   ```

6. **If only "best" checkpoint exists (no training_state.pt):**
   Resume from best eval checkpoint (loses optimizer state, restarts from that step):
   ```bash
   # Same command as above but: --resume-from quantize/runs/v8-qwen3-8b/best
   ```
   Note: "best" was saved via save_pretrained() and DOES contain SVID alpha/beta/layernorm
   keys (verified: 576 keys). The --resume-from code loads them correctly into SVIDBitLinear.

7. **If no checkpoint exists:** Start fresh with the v8 launch command above.
8. Monitor: `tail -f run_v8.log` or `run_v8_resumed.log`
9. Tensorboard: `tensorboard --logdir quantize/runs/v8-qwen3-8b/tensorboard --bind_all`

**IMPORTANT: Current v8 run (process loaded before checkpoint code was added) does NOT save
periodic checkpoints.** Only the "best" checkpoint at step 2000 (1/8 eval) exists. If it crashes,
resume from that checkpoint — you lose steps 2000-current but not everything.
Next launch will save checkpoints every 500 steps with full optimizer state.

**Current best launch command (v8 — full OneBit architecture):**
```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 10000 \
    --max-examples 30000 --epochs 50 --lr 1e-4 \
    --gen-check-interval 200 --eval-interval 1000 \
    --output-dir quantize/runs/v8-qwen3-8b \
    --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 \
    --unlikelihood-weight 0 \
    2>&1 | tee run_v8.log
```

**GPU layout (v5.3):**
- Teacher (4-bit NF4, 6.4 GB) on GPU 0
- Student layers 0-17 + embed on GPU 0 (~8 GB weights + optimizer/grads)
- Student layers 18-35 + norm + lm_head on GPU 1 (~8 GB weights + optimizer/grads)
- Peak: ~50-60 GB per GPU (vs 84 GB OOM when student was on single GPU)
- Uses `accelerate.dispatch_model()` for automatic tensor routing between GPUs

**Go/no-go decision points:**
- Step 75: loss should be < 5.0 and declining
- Step 100: gen check should show English words (not single-token repeat)
- Step 200: gen should show partial answers or meaningful fragments
- Step 500: eval should score ≥ 1/8 (any correct answer = breakthrough)
- If kill signal: stop run, check fallback plan in memory/project_runbook.md

**What has already failed (don't repeat):**
- v4.3: naive sign(w) init → loss converges to 1.9, gen collapses (killed step 300)
- v5.0: GPTQ binary values as weights → flat gradients, worse than v4.3 (killed step 3)
- v5.1: on-policy OOM'd at step 25 — student on single GPU couldn't fit 2 forward passes
- v5.2: killed before results — replaced by v5.3 with split student
- v5.3: unlikelihood too weak (0.1), STE clip too aggressive (1.0), on-policy too slow to ramp
- v5.4: same trajectory as v5.3 despite stronger hyperparameters — problem is structural not tuning
- v6 SVID+500k: loss 2.24 but gen "Okayimport" 0/8 — 5-term loss + no LayerNorm
- v7 SVID+simple loss: still 0/8 — missing LayerNorm inside BitLinear (THE primary bug)
- 5-term loss (MSE+cos+CE+h_MSE+UL) → conflicting gradients → degenerate generation modes
- LR 5e-6 was 80x too low (OneBit uses 4e-4) — sign landscape frozen from initialization
- Vanilla/clipped STE → poor gradient quality. Use tanh-STE.
- Weight init at full FP16 magnitude → 42% gradient at start. Use sign(W)*0.01 → 100%.
- RMS init for alpha/beta → loses covariance structure. Use NMF.
- Last-layer-only hidden MSE → layers 1-35 unconstrained. Use all-layer alignment.
- ProgressiveQuantizedLinear on 8B → OOM. Use SVIDBitLinear only.
- KL divergence → explodes to 3600+. Use normalized MSE + cosine instead.
- Student on single GPU + on-policy → OOM at 84 GB. Must split student across both GPUs.
- 35k examples is 400x too little data. Scale data if current approach fails.

**Fallback plan (if v5.3 fails):**
1. Scale data 100x (synthetic from teacher, OneBit-style: 100k examples, 50 epochs → ~10B tokens)
2. OneBit SVID decomposition: W = sign(W) * outer(a, b)
3. Tanh progressive schedule (BinaryLLM, arXiv 2508.06974)
4. Curriculum bit-width: 4-bit → 2-bit → 1-bit

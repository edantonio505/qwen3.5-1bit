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

# CURRENT best command for 8B (v5: GPTQ init + QAT):
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B \
    --use-4bit-teacher \
    --max-steps 3000 \
    --gen-check-interval 200 \
    --eval-interval 500 \
    --output-dir quantize/runs/v5-qwen3-8b \
    2>&1 | tee run_v5.log

# Previous (v4.3 — loss converges but generation collapses):
# python3 quantize/run_v4.py --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 3000

# Legacy (simpler, no scheduled sampling):
# python3 quantize/run_cloud.py --model Qwen/Qwen3-8B
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

v7 (running) — SVID + simple loss + 30k×50 epochs:
- **Simple loss (OneBit recipe): soft CE + hidden MSE ONLY.** No MSE, no cosine, no hard CE, no UL.
- 30k examples × 50 epochs (not 500k×1 epoch): repetition > diversity for 1-bit sign learning
- On-policy with temperature=0.8, prefix=32 tokens (explores diverse sequences)
- Starting loss: 18.8 (soft CE over 151k vocab — higher number but cleaner signal)
- This is the first run matching what actually worked in published research.

**v5 approach: GPTQ init + on-policy distillation + unlikelihood + clipped STE:**

Six research-backed changes, each tied to a specific paper:

| Change | Paper | arXiv | Key Technique |
|--------|-------|-------|---------------|
| On-policy distillation | MiniLLM | 2306.08543 | Reverse KL + student rollouts (fixes generation collapse) |
| On-policy mix ratio | GKD | 2306.13649 | Tunable on-policy fraction |
| Unlikelihood loss | Unlikelihood Training | 1908.04319 | Penalize repeated tokens during training |
| Clipped STE | PV-Tuning | 2405.14852 | Zero grad for weights far from decision boundary |
| Multi-layer distillation | TinyBERT | 1909.10351 | Match hidden states at layers 7,15,23,31 |
| Hidden state distill (last layer) | BitDistill | 2510.13998 | Original motivation for hidden state matching |
| Block-wise QAT | EfficientQAT | 2407.11062 | Freeze/unfreeze blocks to reduce VRAM |
| Activation-weighted GPTQ | AWQ | 2306.00978 | Weight sign-flip priority by activation magnitude |
| GPTQ init | "What Makes Low-Bit QAT Work" | 2601.14888 | Hessian-optimal binary weights as QAT starting point |
| Hadamard rotation | QuEST / QuIP# | 2502.05003 / 2402.04396 | Spread outlier energy before binarization |
| Binary LLM feasibility | FBI-LLM | 2407.07093 | Proves {-1,+1} LLMs work at 7B scale |

**CRITICAL INSIGHT: Simple loss > complex loss for 1-bit (v6/v7 finding):**
FBI-LLM proved that distillation-only soft CE outperforms combined losses. OneBit uses just
soft CE + hidden state MSE. Our 5-term loss (MSE+cos+CE+h_MSE+UL) caused conflicting gradients —
the model found degenerate modes that minimized some terms while ignoring others (v4.3: `\n\n`,
v5.3: `, 01.`, v6: `Okayimport`). v7 uses ONLY soft CE + hidden MSE (OneBit recipe).

**CRITICAL INSIGHT: Repetition > diversity for 1-bit:**
At 1-bit, each weight is a single sign bit. The optimizer can only flip signs. To correctly
set 8B sign bits, each example must be seen MANY times (OneBit: 50 epochs). Seeing 500k unique
examples once (v6) is worse than seeing 30k examples ~5 times (v7). Repetition hammers
the gradient signal to flip signs correctly.

**On-policy distillation (MiniLLM, arXiv 2306.08543):**
Standard KD uses forward KL → mode averaging → incoherent generation. On-policy uses student-
generated sequences with temperature sampling (0.8) from short prefixes (32 tokens), forcing
the model to practice recovery from its own errors. 15% of training steps.

**GPTQ init fix (v5.1):** Keep FP16 magnitudes, only flip signs to match GPTQ Hessian-optimal.
Binary values as weights give flat gradient landscape (v5.0 finding).

**Data volume:** OneBit used 13.5B tokens (132k × 2048 × 50 epochs). v7 uses 30k × 512 × 50
epochs ≈ 768M tokens. Still less but with repetition emphasis matching OneBit's approach.

**PrismML Bonsai (reference target):**
- Built from Qwen3-8B (standard dense transformer, NOT Qwen3.5 hybrid)
- True binary {-d, +d}, Q1_0_g128 applied to ALL layers including embed + LM head
- Uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon)
- 1-bit Bonsai 8B: 1.15 GB, 70.5 avg benchmark, 5-8x faster inference
- Whitepaper: `1-bit-bonsai-8b-whitepaper.pdf`

### Quantization Files

```
quantize/
├── run_v5.py          # CURRENT: v5 — GPTQ init + QAT with hidden state distillation
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

**Phase 2 — QAT Fine-tuning (v7 — current best, multi-GPU):**
- **Quantizer:** SVIDBitLinear (OneBit SVID): `w_q = sign(w) * alpha_i * beta_j` per layer
- **Init:** FP16 magnitudes preserved, signs flipped to match GPTQ Hessian-optimal
- **Loss:** `--simple-loss` — soft CE (teacher probs as targets) + hidden state MSE (OneBit recipe)
  - NO normalized MSE, NO cosine, NO hard CE, NO unlikelihood
  - FBI-LLM proved single-objective distillation outperforms multi-term losses
- **On-policy:** 15% of steps, temp=0.8 sampling from 32-token prefix (explores diverse sequences)
- **Data:** 30k examples × 50 epochs — repetition > diversity for sign bit learning
- **Teacher:** Frozen 4-bit NF4 on GPU 0 (shared with student layers 0-17)
- **Student:** Split across both GPUs via accelerate.dispatch_model()
- **Skipped layers:** Embedding + LM head kept in FP16
- **Optimizer:** 8-bit AdamW, 10x LR for SVID alpha/beta vectors

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

**When resuming on a new server:**
1. Check GPU setup: `nvidia-smi` — need 2x 80GB+ GPUs
2. Check deps: `python3 -c "import torch, transformers, bitsandbytes, accelerate; print('OK')"`
3. Check if GPTQ checkpoint exists: `ls quantize/runs/v5-qwen3-8b/gptq_checkpoint/group_scales.pt`
4. If yes: launch with `--skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint`
5. If no: launch without those flags (runs ~15 min GPTQ Phase 1 first)
6. Monitor: `tail -f run_v5.3.log`

**Current best launch command (v7 — OneBit recipe):**
```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 10000 \
    --max-examples 30000 --epochs 50 \
    --gen-check-interval 200 --eval-interval 1000 \
    --output-dir quantize/runs/v7-qwen3-8b \
    --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 \
    --unlikelihood-weight 0 \
    2>&1 | tee run_v7.log
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
- v6 SVID+500k: loss dropped to 2.24 but gen "Okayimport" attractor, 0/8. 5-term loss is the problem.
- 5-term loss (MSE+cos+CE+h_MSE+UL) → conflicting gradients → degenerate generation modes
- 500k examples seen once < 30k examples seen 5+ times (repetition matters for 1-bit)
- ProgressiveQuantizedLinear on 8B → OOM. Use BitLinear/SVIDBitLinear only.
- KL divergence → explodes to 3600+. Use normalized MSE + cosine instead.
- Student on single GPU + on-policy → OOM at 84 GB. Must split student across both GPUs.
- 35k examples is 400x too little data. Scale data if current approach fails.

**Fallback plan (if v5.3 fails):**
1. Scale data 100x (synthetic from teacher, OneBit-style: 100k examples, 50 epochs → ~10B tokens)
2. OneBit SVID decomposition: W = sign(W) * outer(a, b)
3. Tanh progressive schedule (BinaryLLM, arXiv 2508.06974)
4. Curriculum bit-width: 4-bit → 2-bit → 1-bit

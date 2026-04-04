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

### IMMEDIATE NEXT STEP — Run on 2× GPU Server

The pipeline is validated on 1.7B. Now needs Qwen3-8B on a multi-GPU server.

**CRITICAL: Single A100 80GB is NOT enough.** We tried extensively on 2026-04-04 — the 8B model OOMs even with batch=1, seq=512, 8-bit Adam, and memory-optimized forward pass. The fundamental issue: teacher (6 GB) + student weights (16.5 GB) + gradients (16.5 GB) + optimizer states (16-32 GB) + forward pass intermediate tensors (~16.5 GB from ProgressiveQuantizedLinear w/w_1bit/w_eff) = ~88 GB minimum.

**The code now supports multi-GPU: teacher on GPU 0, student on GPU 1.** Auto-detected via `auto_config()`.

```bash
# One command on a 2-GPU server (2× A100 80GB, 2× A40 48GB, etc.):
git clone https://github.com/edantonio505/qwen3.5-1bit.git
cd qwen3.5-1bit
bash setup_and_run_8b.sh
```

Or manually:
```bash
python3 -m venv .venv

# Install PyTorch matching your CUDA version first
# Check CUDA: nvidia-smi | head -3
# CUDA 12.4: pip install torch --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.8+: pip install torch (latest should work)
.venv/bin/pip install torch transformers accelerate datasets bitsandbytes sentencepiece protobuf huggingface-hub

PYTHONUNBUFFERED=1 .venv/bin/python quantize/run_v4.py \
  --model Qwen/Qwen3-8B \
  --use-4bit-teacher \
  --max-steps 3000 \
  --gen-check-interval 200 \
  --eval-interval 500 \
  --output-dir quantize/runs/v4.3-qwen3-8b \
  2>&1 | tee run.log
```

### Quantization Format: Q1_0_g128

Every weight is binary: `w_i = scale_g * (2*bit_i - 1)` where `bit_i ∈ {0,1}` and `scale_g` is a shared FP16 scale per group of 128 weights. This is the format PrismML uses for Bonsai models.

### What We Learned (Critical Knowledge — Read ALL of This)

**Use Qwen3-8B (standard transformer). NOT Qwen3.5 (hybrid).**
- Qwen3.5 uses GatedDeltaNet (75% linear attention with recurrent state) — fundamentally harder to quantize. Recurrent state compounds errors through time.
- Qwen3 models are pure standard transformers — what PrismML used.
- Training is 6x faster on standard transformers.
- 1.7B was tested and works but CE plateaus at ~1.7. Likely needs 8B scale for the breakthrough.

**PTQ (post-training quantization) does NOT work at 1-bit:**
- GPTQ with Hessian compensation, Hadamard rotation, sign-flip refinement → 0% on both 2B and 8B
- Error compounds catastrophically through transformer layers — dead after 5 layers
- `gptq_1bit.py` has `--eval-every N` flag for early abort detection
- DO NOT waste time on PTQ approaches. QAT is required.

**QAT (quantization-aware training) is required, with these specific techniques:**
- KL divergence explodes over 151k vocab at 1-bit — use normalized MSE + cosine + CE instead
- Teacher forcing causes generation collapse (loss converges but model outputs `\n\n`) — need scheduled sampling
- Jumping straight to 1-bit fails — need progressive quantization (AggressiveQuantizer: 0.5→1.0)
- Word doubling artifact ("TheThe", "is is") at 1-bit — use repetition_penalty=2.0 + no_repeat_ngram_size=3
- Embed + LM head must stay in FP16 (critical for generation quality)
- Training data must include short-answer QA (TriviaQA + GSM8K + custom factual pairs), not just conversations

**Memory optimizations already implemented (critical for 8B):**
- 8-bit AdamW via bitsandbytes — saves ~48 GB optimizer memory (FP32 → INT8 momentum+variance)
- Memory-optimized `ProgressiveQuantizedLinear.forward()`: at noise_scale=1.0 (80% of training), only `w_1bit` is computed, no blending. During warmup, uses `w + noise*(w_1bit-w)` with immediate `del w_1bit`
- Teacher activations are `.detach()`ed and cache is cleared before student forward
- OOM failsafe: aborts after 10 consecutive OOMs instead of infinite retry loop
- Even with ALL these optimizations, single A100 80GB still OOMs for 8B. Multi-GPU is required.

**PrismML Bonsai (reference target):**
- Built from Qwen3-8B (standard dense transformer, NOT Qwen3.5 hybrid)
- True binary {-d, +d}, Q1_0_g128 applied to ALL layers including embed + LM head
- Uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon)
- 1-bit Bonsai 8B: 1.15 GB, 70.5 avg benchmark, 5-8x faster inference
- Whitepaper: `1-bit-bonsai-8b-whitepaper.pdf`

### Quantization Files

```
quantize/
├── run_v4.py          # CURRENT: QAT v4.3 (progressive quant + scheduled sampling + QA data mix)
├── run_cloud.py       # Legacy: QAT with 4-bit teacher, auto-detects GPUs/VRAM
├── run.py             # Legacy: Local QAT with SubLN + dynamic scales
├── gptq_1bit.py       # GPTQ 1-bit PTQ (proven insufficient, useful for analysis)
├── quantize_lib.py    # Shared library: ProgressiveQuantizedLinear, Hadamard, learned scales, STE
├── auto_tune.py       # Hyperparameter search loop
├── evaluate.py        # Benchmark evaluation
├── export_gguf.py     # Export to Q1_0_g128 GGUF format
└── train.py           # Standalone training script
```

### v4.3 Architecture (Current)

Four key techniques combined:
1. **AggressiveQuantizer** — starts noise at 0.5, ramps to 1.0 by 20% of steps, stays at 1.0 for 80%. Maximizes training time at full 1-bit.
2. **Scheduled sampling** — mixes student's own predictions into training inputs (10%→30% over training) to fix exposure bias from teacher forcing.
3. **Normalized MSE + cosine + CE loss** — distills from BF16/4-bit teacher without KL explosion. Weights: MSE 0.4, cosine 0.2, CE 0.4.
4. **QA data mix** — 60% OpenHermes conversations + 40% short-answer QA (TriviaQA + GSM8K + custom factual pairs). Teaches model to produce concise answers, not paragraphs.

Uses `ProgressiveQuantizedLinear` from `quantize_lib.py` with learned `log_scale` parameters (10x LR multiplier).
Eval uses `repetition_penalty=2.0` + `no_repeat_ngram_size=3` to counter word-doubling artifact.

### GPU Requirements

| Model | Task | Min VRAM | Notes |
|-------|------|----------|-------|
| Qwen3-1.7B | QAT training | 40 GB | A40 48GB works. CE plateaus at ~1.7. |
| **Qwen3-8B** | **QAT training** | **2× 40GB+** | **Needs 2 GPUs: teacher on GPU 0, student on GPU 1. Single 80GB OOMs.** |
| Qwen3-8B | GPTQ (PTQ) | 20 GB | Don't bother — PTQ doesn't work at 1-bit |

**Why single GPU fails for 8B QAT:** ProgressiveQuantizedLinear's forward pass creates `w`, `w_1bit`, and `w_eff` tensors — 3× student weight memory (~49.5 GB) during forward. Combined with teacher, optimizer states, and gradients, peak memory exceeds 80 GB even at batch=1.

**Multi-GPU solution:** Code auto-detects 2+ GPUs in `auto_config()` and places teacher on `cuda:0` (~6 GB), student on `cuda:1` (~59 GB with optimizer+gradients). Only logits (~300 MB) transfer between GPUs per step.

### Complete Results History

| Approach | Model | Score | CE at 1-bit | Key Finding |
|----------|-------|-------|-------------|-------------|
| GPTQ PTQ | 2B/8B | 0% | N/A | PTQ dead end — error compounds through layers |
| QAT run_cloud.py | 2B | 0% | ~0.003 train | Loss converges but generation collapses (teacher forcing) |
| QAT v4.0 (top-K KL) | 2B hybrid | 0% | 3.5 | KL clamped at 50, drowned CE signal |
| QAT v4.1 (MSE+cos, 1000 steps) | 2B hybrid | 0% | 1.8 | First contextual English ("The sun is a warm") |
| QAT v4.2 on Qwen3.5-2B | 2B hybrid | 12% | 0.73* | *noise=0.94 not full 1-bit. Hybrid arch is blocker. |
| QAT v4.2 on Qwen3-1.7B (chat only) | 1.7B standard | 0% | 1.7 plateau | Standard transformer confirmed 6x faster |
| QAT v4.3 on Qwen3-1.7B (QA mix) | 1.7B standard | 12% | ~1.7 plateau | QA data didn't change CE trajectory at 1.7B scale |
| QAT v4.3 on Qwen3-8B (1× A100) | 8B standard | OOM | N/A | Single 80GB GPU cannot fit 8B QAT. Needs 2 GPUs. |
| **QAT v4.3 on Qwen3-8B (2× GPU)** | **8B standard** | **TBD** | **TBD** | **NEXT: Run on 2× A100/A40 server** |

### What to Watch For During 8B Training

- **CE at 1-bit entry (step ~600)**: Should be ~2.0. If much higher, increase LR.
- **CE at step 1000**: If below 1.5, we're on track. If plateauing at 1.7+ like 1.7B, may need longer training.
- **Generation checks**: Look for factual content, not just English words. "Paris" for France, numbers for math.
- **Step 500 eval**: First full eval. If >12%, the approach is working.
- **Step 1000 eval**: If >25%, this is a breakthrough. Scale up training.

### If 8B Also Plateaus at CE ~1.7

Fallback options (in priority order):
1. **Much longer training** — 10,000+ steps at full 1-bit with LR restart
2. **Ternary {-1, 0, +1}** — BitNet b1.58 approach, more expressive than binary
3. **Attention distillation** — match teacher's attention patterns, not just output logits
4. **Layer-wise progressive** — quantize one layer at a time instead of all at once

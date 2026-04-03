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

### Quantization Format: Q1_0_g128

Every weight is binary: `w_i = scale_g * (2*bit_i - 1)` where `bit_i ∈ {0,1}` and `scale_g` is a shared FP16 scale per group of 128 weights. This is the format PrismML uses for Bonsai models.

### What We Learned (Critical Knowledge)

**PTQ (post-training quantization) does NOT work at 1-bit:**
- GPTQ with Hessian compensation, Hadamard rotation, sign-flip refinement → 0% on both 2B and 8B
- Error compounds catastrophically through transformer layers — dead after 5 layers
- `gptq_1bit.py` has `--eval-every N` flag for early abort detection

**QAT (quantization-aware training) is required, with specific techniques:**
- KL divergence explodes over 151k vocab at 1-bit — use normalized MSE + cosine + CE instead
- Teacher forcing causes generation collapse (loss converges but model outputs `\n\n`) — need scheduled sampling
- Jumping straight to 1-bit fails — need progressive quantization (gradual noise annealing)
- Word doubling artifact ("TheThe", "is is") at 1-bit — repetition penalty 1.3 in generation reveals hidden knowledge
- Embed + LM head must stay in FP16 (critical for generation quality)

**PrismML Bonsai (reference target):**
- Built from Qwen3-8B (standard dense transformer, NOT Qwen3.5 hybrid)
- True binary {-d, +d}, Q1_0_g128 applied to ALL layers including embed + LM head
- Uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon)
- 1-bit Bonsai 8B: 1.15 GB, 70.5 avg benchmark, 5-8x faster inference
- Whitepaper: `1-bit-bonsai-8b-whitepaper.pdf`

### Quantization Files

```
quantize/
├── run_v4.py          # Current: QAT v4.2 (progressive quant + scheduled sampling + MSE distillation)
├── run_cloud.py       # Prior: QAT with 4-bit teacher, auto-detects GPUs/VRAM
├── run.py             # Prior: Local QAT with SubLN + dynamic scales
├── gptq_1bit.py       # GPTQ 1-bit PTQ (proven insufficient, but useful for analysis)
├── quantize_lib.py    # Shared library: ProgressiveQuantizedLinear, Hadamard, learned scales, STE
├── auto_tune.py       # Hyperparameter search
├── evaluate.py        # Benchmark evaluation
├── export_gguf.py     # Export to Q1_0_g128 GGUF format
└── train.py           # Standalone training script
```

### Running Quantization (Current Approach: run_v4.py)

```bash
# Install deps (if no venv)
python3 -m venv .venv
.venv/bin/pip install torch transformers accelerate datasets sentencepiece protobuf huggingface-hub

# Quick validation (300 steps, ~2 hours)
.venv/bin/python quantize/run_v4.py --model Qwen/Qwen3.5-2B --max-steps 300

# Full training (3000 steps, ~20 hours on A40)
.venv/bin/python quantize/run_v4.py --model Qwen/Qwen3.5-2B

# 8B model (needs A100 80GB+)
.venv/bin/python quantize/run_v4.py --model Qwen/Qwen3-8B --use-4bit-teacher
```

### v4.2 Architecture (Current)

Three key techniques combined:
1. **AggressiveQuantizer** — starts noise at 0.5, ramps to 1.0 by 20% of steps, stays at 1.0 for 80%. Maximizes training time at full 1-bit.
2. **Scheduled sampling** — mixes student's own predictions into training inputs (10%→30% over training) to fix exposure bias from teacher forcing.
3. **Normalized MSE + cosine + CE loss** — distills from BF16 teacher without KL explosion. Weights: MSE 0.4, cosine 0.2, CE 0.4.

Uses `ProgressiveQuantizedLinear` from `quantize_lib.py` with learned `log_scale` parameters (10x LR multiplier).

### GPU Requirements

| Model | Task | Min VRAM | Notes |
|-------|------|----------|-------|
| Qwen3.5-2B | QAT training | 40 GB | A40 48GB works |
| Qwen3-8B | QAT training | 78 GB | A100 80GB with 4-bit teacher |
| Qwen3-8B | GPTQ (PTQ) | 20 GB | A40 works, but PTQ doesn't help at 1-bit |

### Results History

| Approach | Model | Score | Key Output |
|----------|-------|-------|------------|
| GPTQ PTQ | 2B/8B | 0% | "FGFG" garbage |
| QAT run_cloud.py | 2B | 0% | `\n\n\n` empty |
| QAT v4.0 (top-K KL) | 2B | 0% | English words but KL drowned CE |
| QAT v4.1 (MSE+cos, 1000 steps) | 2B | 0% | "The sun is a warm" — contextual! |
| QAT v4.2 baseline (untrained + rep penalty) | 2B | 12% | 1/8 correct answers |
| QAT v4.2 (3000 steps, in progress) | 2B | TBD | CE: 9.3→0.03→0.68 stable at noise=0.87 |

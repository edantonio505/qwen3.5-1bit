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

# CURRENT best command for 8B (v4.3 with BitLinear):
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v4.py \
    --model Qwen/Qwen3-8B \
    --use-4bit-teacher \
    --max-steps 3000 \
    --gen-check-interval 200 \
    --eval-interval 500 \
    --output-dir quantize/runs/v4.3-qwen3-8b \
    2>&1 | tee run.log

# Legacy (simpler but less features):
PYTHONUNBUFFERED=1 python quantize/run_cloud.py --model Qwen/Qwen3-8B 2>&1 | tee run.log
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

**2x A100 80GB multi-GPU setup (confirmed working):**
- Teacher (4-bit NF4) on GPU 0: ~6.4 GB
- Student (BitLinear) on GPU 1: ~16.5 GB → peaks at ~78 GB during backward
- Config: batch=1, seq=512, grad_accum=16, 8-bit AdamW
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with fragmentation
- Teacher logits `.detach().to(student_device)` — cross-GPU transfer, then free teacher KV cache

**Single A100 80GB is NOT enough for 8B QAT:**
- Even with BitLinear, student alone peaks at ~78 GB during backward
- Teacher needs another ~6 GB on a separate GPU
- Use 160GB+ single GPU or 2x 40GB+ multi-GPU

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

**PrismML Bonsai (reference target):**
- Built from Qwen3-8B (standard dense transformer, NOT Qwen3.5 hybrid)
- True binary {-d, +d}, Q1_0_g128 applied to ALL layers including embed + LM head
- Uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon)
- 1-bit Bonsai 8B: 1.15 GB, 70.5 avg benchmark, 5-8x faster inference
- Whitepaper: `1-bit-bonsai-8b-whitepaper.pdf`

### Quantization Files

```
quantize/
├── run_v4.py          # CURRENT: v4.3 QAT with BitLinear (8B), progressive (2B), scheduled sampling
├── run_cloud.py       # Cloud QAT with BitLinear + 4-bit teacher, auto-detects GPUs/VRAM/arch
├── run.py             # Local QAT with SubLN + dynamic scales (for DIGITS)
├── quantize_lib.py    # Shared library: ProgressiveQuantizedLinear, Hadamard, learned scales, STE
├── auto_tune.py       # Hyperparameter search loop (8 configs)
├── evaluate.py        # Benchmark evaluation
├── export_gguf.py     # Export to Q1_0_g128 GGUF format
└── train.py           # Standalone training script
```

### Training Architecture (run_v4.py)

- **Loss:** Normalized logit MSE (0.4) + cosine similarity (0.2) + CE (0.4)
- **Teacher:** Frozen copy of base model (4-bit via bitsandbytes for 8B+)
- **Student:** Same model with BitLinear layers (1-bit weights, learned group scales)
- **Quantizer:** BitLinear for 8B (memory-efficient, int8 signs), ProgressiveQuantizedLinear for 2B (blending schedule)
- **Scheduled sampling:** 10%→30% of tokens replaced with student's own predictions during training
- **Data mix:** 60% OpenHermes chat + 40% QA (TriviaQA + GSM8K + custom factual)
- **Skipped layers:** Embedding + LM head kept in FP16
- **Optimizer:** 8-bit AdamW with separate LR for scale params (10x multiplier)
- **Eval:** Repetition penalty 2.0, no_repeat_ngram_size=3, greedy decoding

### GPU Requirements

| Model | Config | Total VRAM | Example GPU |
|-------|--------|-----------|-------------|
| Qwen3.5-2B | BF16 teacher + ProgressiveQuantized student | ~40 GB | A40 48GB |
| Qwen3-8B | 4-bit teacher (GPU 0) + BitLinear student (GPU 1) | ~78 GB peak per GPU | 2x A100 80GB |
| Qwen3-8B | 4-bit teacher + BitLinear student (single GPU) | ~85 GB peak | 1x H100 96GB or 1x A100 160GB |
| Qwen3.5-35B | 4-bit teacher + student | ~380 GB | 8x A100 80GB |

**A single 80GB GPU (A100) will OOM on 8B.** Use 2x A100 80GB (teacher/student split) or 1x 160GB+.

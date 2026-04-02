# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bonsai Demo is a local inference runner for PrismML's 1-bit quantized language models (8B, 4B, 1.7B). It supports two backends:
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

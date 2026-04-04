# Qwen 1-Bit

Experimental 1-bit quantization-aware training (QAT) for Qwen language models. The goal is to compress Qwen3/3.5 models to true 1-bit weights ({-d, +d} per group of 128) while preserving generation quality.

Inspired by [PrismML's Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo), which demonstrated that a 1-bit 8B model (1.15 GB) can score competitively against full-precision models (16+ GB). This repo includes PrismML's demo scripts for running their pre-built Bonsai models, plus our own QAT training pipeline.

## Status

**Active development.** Training loss converges but autoregressive generation still collapses. Moving to cloud GPUs (RunPod) for more compute. See [Findings](#findings) below.

## Quick Start — Training

```bash
# Install dependencies
pip install torch transformers datasets accelerate sentencepiece protobuf huggingface-hub

# Local training (small model)
PYTHONUNBUFFERED=1 python quantize/run_cloud.py --model Qwen/Qwen3.5-2B

# Quick validation (300 steps, ~2 hours)
PYTHONUNBUFFERED=1 python quantize/run_cloud.py --model Qwen/Qwen3.5-2B --max-examples 3000

# 8B model on cloud GPU (160GB+ recommended)
git clone https://github.com/edantonio505/qwen3.5-1bit.git && cd qwen3.5-1bit
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# Or manually:
PYTHONUNBUFFERED=1 python quantize/run_cloud.py --model Qwen/Qwen3-8B

# Legacy approaches (for reference)
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
| Quantization | STE (straight-through estimator) with binary {-d, +d} weights via `ProgressiveQuantizedLinear` |
| Noise schedule | AggressiveQuantizer: start noise=0.5, ramp to 1.0 by 20% of steps, full 1-bit for 80% |
| Scales | Learned per group of 128 (log_scale parameters, 10x LR multiplier) |
| Distillation | Normalized logit MSE (0.4) + cosine similarity (0.2) + CE (0.4). NOT KL — it explodes at 1-bit |
| Scheduled sampling | Mix student's own predictions into training inputs (10%→30%) to fix exposure bias |
| Teacher | Frozen BF16 copy (or 4-bit via bitsandbytes for 8B models) |
| Skipped layers | Embedding + LM head kept in FP16 (critical for generation) |
| Eval | Repetition penalty 1.3 to counter word-doubling artifact |

### Key Files

```
quantize/
├── run_cloud.py      # Cloud training (multi-GPU, 4-bit teacher, auto-detect)
├── gptq_1bit.py      # GPTQ 1-bit PTQ (proven insufficient, useful for analysis)
├── run.py            # Legacy: local training with SubLN + dynamic scales
├── run_cloud.py      # Legacy: cloud training (multi-GPU, 4-bit teacher)
├── quantize_lib.py   # Core: ProgressiveQuantizedLinear, STE, Hadamard, learned scales
├── auto_tune.py      # Hyperparameter search loop
├── evaluate.py       # Benchmark evaluation
├── export_gguf.py    # Export to Q1_0_g128 GGUF format
└── train.py          # Standalone training script
```

### Hardware Requirements

Measured from actual training runs (teacher + student + optimizer + gradients + activations):

| Model | Config | Total VRAM | Example GPU |
|---|---|---|---|
| Qwen3.5-2B | BF16 teacher + student | ~40 GB | A40 48GB |
| Qwen3-8B | 4-bit teacher + BF16 student | ~100 GB peak | 1x 160GB or 2x48GB |
| Qwen3.5-35B | 4-bit teacher + BF16 student | ~380 GB | 8x A100 80GB |

**A single 80GB GPU (A100) will OOM on 8B.** Peak memory during backward exceeds 100GB.

**Note:** 24GB GPUs (RTX 3090/4090) cannot fit even the 2B model due to optimizer states and activations.

---

## Findings

### Key Discoveries

1. **KL divergence explodes at 1-bit** — over 151k vocab, KL goes from 16 to 3600+. Clamping makes it useless. Use normalized MSE + cosine similarity instead.

2. **Teacher forcing causes generation collapse** — training loss converges perfectly (CE→0.003, cosine→0.30) but the model outputs `\n\n` repeated 60 times during generation. The model never learns to recover from its own errors.

3. **Single A100 80GB is NOT enough for 8B QAT** — teacher + student + optimizer + activations peaks at ~100-130 GB. OOMs even at batch=1. Use 160GB+ GPU or 4-bit teacher to fit.

4. **SubLN helps at initialization** — RMSNorm before each binary linear prevents hidden state collapse initially, but training pushes the model back to the `\n\n` attractor.

5. **The model IS learning** — "Paris" moves from rank 134,940 to 18,016 in the logit ranking after 375 steps. It needs to reach rank 1 out of 151,669 tokens. More steps and more data should help.

6. **Embed + LM head should stay FP16** — these layers directly interface with the token space. Quantizing them breaks generation immediately.

### What Works
- Normalized MSE + cosine + CE distillation is stable (no explosion)
- Learned per-group scales with 10x LR multiplier
- 4-bit teacher via bitsandbytes saves ~12 GB VRAM
- Training loss converges consistently across all runs

### What Doesn't Work Yet
- Autoregressive generation still collapses to `\n\n` after training
- Need more compute (longer training, bigger GPUs) to test whether more steps break through

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

- [PrismML Bonsai-8B Whitepaper](1-bit-bonsai-8b-whitepaper.pdf)
- [BitNet b1.58 — The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- [OneBit — Towards Extremely Low-bit LLMs](https://arxiv.org/abs/2402.11295)
- [BiLLM — Pushing the Limit of PTQ for LLMs](https://arxiv.org/abs/2402.04291)
- [BitDistiller — Sub-4-Bit LLM Self-Distillation](https://arxiv.org/abs/2402.10631)
- [QuIP# — Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)
- [Optimal Brain Surgeon (Hassibi, 1993)](https://papers.nips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)

## License

Apache 2.0 (inherited from PrismML's Bonsai-demo).

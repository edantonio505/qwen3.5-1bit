# Qwen 1-Bit

Experimental 1-bit quantization-aware training (QAT) for Qwen language models. The goal is to compress Qwen3/3.5 models to true 1-bit weights ({-d, +d} per group of 128) while preserving generation quality.

Inspired by [PrismML's Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo), which demonstrated that a 1-bit 8B model (1.15 GB) can score competitively against full-precision models (16+ GB). This repo includes PrismML's demo scripts for running their pre-built Bonsai models, plus our own QAT training pipeline.

## Status

**Active development.** Multiple quantization approaches tested. Current approach (v4.2) uses progressive quantization + scheduled sampling + MSE distillation. The model generates contextually relevant English at 1-bit ("The sun is a warm" for sky-related prompts) and scores 12% on simple QA with repetition penalty. Full training run (3000 steps) in progress with CE stabilized at ~0.68 at noise=0.87 — much better than prior attempts. See [Findings](#findings) below.

## Quick Start — Training

```bash
# Install dependencies
pip install torch transformers datasets accelerate sentencepiece protobuf huggingface-hub

# Current approach: v4.2 (progressive quant + scheduled sampling)
PYTHONUNBUFFERED=1 python quantize/run_v4.py --model Qwen/Qwen3.5-2B

# Quick validation (300 steps, ~2 hours)
PYTHONUNBUFFERED=1 python quantize/run_v4.py --model Qwen/Qwen3.5-2B --max-steps 300

# 8B model (needs A100 80GB+)
PYTHONUNBUFFERED=1 python quantize/run_v4.py --model Qwen/Qwen3-8B --use-4bit-teacher

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

### Training Approach (v4.2 — Current)

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
├── run_v4.py         # Current: QAT v4.2 (progressive quant + scheduled sampling + MSE)
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
| Qwen3-8B | 4-bit teacher + BF16 student | ~78 GB | A100 80GB or 2x48GB |
| Qwen3.5-35B | 4-bit teacher + BF16 student | ~380 GB | 8x A100 80GB |

**Note:** 24GB GPUs (RTX 3090/4090) cannot fit even the 2B model due to optimizer states and activations.

---

## Findings

### Approach Evolution & Results

| Approach | Score | Generation Output | Key Issue |
|----------|-------|-------------------|-----------|
| GPTQ PTQ (gptq_1bit.py) | 0% | "FGFG" random garbage | PTQ fundamentally insufficient at 1-bit |
| QAT v1-v3 (run_cloud.py, run.py) | 0% | `\n\n\n` empty | Exposure bias from teacher forcing |
| QAT v4.0 (top-K KL loss) | 0% | English words ("the", "higher") | KL clamped at 50, drowned CE signal |
| QAT v4.1 (MSE+cos, 1000 steps) | 0% | Contextual sentences ("The sun is a warm") | Only 300 steps at full 1-bit |
| QAT v4.2 baseline (untrained + rep penalty) | 12% | 1/8 correct | Word doubling hid correct answers |
| QAT v4.2 on Qwen3.5-2B (killed step 525) | 12% | CE=0.73 at noise=0.94 | Hybrid architecture is a blocker |
| QAT v4.2 on Qwen3-1.7B (in progress) | TBD | Standard transformer, 6x faster | Architecture matches PrismML's choice |

### Key Discoveries

1. **Use standard transformers, not hybrid architectures** — Qwen3.5-2B (GatedDeltaNet hybrid) is fundamentally harder to quantize than Qwen3-1.7B/8B (standard transformer). The recurrent state in linear attention compounds quantization errors. PrismML chose standard transformer deliberately. Training is 6x faster on standard transformers.

2. **PTQ cannot handle 1-bit** — GPTQ with Hessian compensation, Hadamard rotation, and sign-flip refinement all fail. Error compounds catastrophically through layers (dead after 5/36 layers). Confirmed on both Qwen3.5-2B and Qwen3-8B.

2. **KL divergence explodes at 1-bit** — over 151k vocab, KL goes to 3600+. Even top-K KL (K=128) clamped at 50 permanently. Use normalized MSE + cosine instead.

3. **Teacher forcing causes generation collapse** — model learns perfect next-token prediction (CE→0.003) but outputs `\n\n` during generation because it never sees its own errors. Fix: scheduled sampling (mix student predictions into training inputs).

4. **Progressive quantization prevents initialization shock** — jumping straight to 1-bit destroys the model. Annealing noise from 0.5→1.0 lets the model adapt gradually. CE stays 2x lower than non-progressive at same noise levels.

5. **Word doubling at 1-bit** — the model generates every word twice ("TheThe", "is is", "world world"). Adding repetition_penalty=1.3 reveals the model actually has correct knowledge hidden behind the doubling pattern.

6. **Most training time should be at full 1-bit** — v4.1 wasted 700/1000 steps on noise 0.0-0.9. v4.2 starts at noise=0.5 and reaches 1.0 by 20% of steps, giving 80% of training at full 1-bit.

### What Works
- Progressive quantization (noise annealing from 0.5→1.0) keeps CE stable
- Normalized MSE + cosine + CE distillation is stable and effective
- Scheduled sampling (10-30% of tokens replaced with student predictions)
- Learned per-group scales with 10x LR multiplier
- Repetition penalty 1.3 in eval reveals hidden knowledge
- Model generates contextually relevant English at full 1-bit after v4.1+

### What Doesn't Work Yet
- Generation accuracy: 0% without repetition penalty, 12% with (untrained baseline)
- Word doubling artifact not fully solved
- Need more training steps at full 1-bit (v4.2 in progress with 2400 steps at noise=1.0)

### Why This Is Hard
PrismML's Bonsai uses proprietary Caltech IP (Babak Hassibi, inventor of Optimal Brain Surgeon). Their approach is described as "mathematically grounded advances designed to preserve reasoning quality under aggressive compression." No research paper has been published.

### Potential Next Steps (if v4.2 plateaus)
1. Try Qwen3-8B (standard transformer, more redundant — what PrismML actually used) on A100 80GB
2. Ternary {-1, 0, +1} quantization (BitNet b1.58) — more expressive
3. Start from v4.1/v4.2 checkpoint instead of fresh weights
4. Attention-level distillation — match attention maps, not just output logits

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

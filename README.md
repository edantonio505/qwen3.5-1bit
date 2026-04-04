# Qwen 1-Bit

Experimental 1-bit quantization-aware training (QAT) for Qwen language models. The goal is to compress Qwen3/3.5 models to true 1-bit weights ({-d, +d} per group of 128) while preserving generation quality.

Inspired by [PrismML's Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo), which demonstrated that a 1-bit 8B model (1.15 GB) can score competitively against full-precision models (16+ GB). This repo includes PrismML's demo scripts for running their pre-built Bonsai models, plus our own QAT training pipeline.

## Status

**Active: v4.3 training run in progress on 2x A100 80GB.** Training Qwen3-8B to true 1-bit (Q1_0_g128) with BitLinear + 4-bit teacher distillation. 3000-step run with scheduled sampling and mixed QA+chat data. Previous runs proved loss converges; this run tests whether BitLinear + more data + scheduled sampling breaks through the generation collapse barrier.

## Quick Start — Training

```bash
# Install dependencies
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# 8B model — CURRENT BEST (v4.3, requires 2x 80GB or 1x 160GB GPU)
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v4.py \
    --model Qwen/Qwen3-8B \
    --use-4bit-teacher \
    --max-steps 3000 \
    --gen-check-interval 200 \
    --eval-interval 500 \
    --output-dir quantize/runs/v4.3-qwen3-8b \
    2>&1 | tee run.log

# 2B model — quick validation (~2 hours on 48GB GPU)
PYTHONUNBUFFERED=1 python3 quantize/run_v4.py --model Qwen/Qwen3.5-2B --max-steps 300

# Legacy cloud script (simpler, fewer features):
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

### Training Approach (v4.3)

| Component | Technique |
|---|---|
| Quantization (8B) | `BitLinear` — STE with int8 sign storage (50% less memory), pure 1-bit from start |
| Quantization (2B) | `ProgressiveQuantizedLinear` — blends FP and 1-bit via noise schedule (0.5→1.0) |
| Scales | Learned per group of 128 (`log_scale` parameters, 10x LR multiplier) |
| Distillation | Normalized logit MSE (0.4) + cosine similarity (0.2) + CE (0.4). NOT KL — it explodes at 1-bit |
| Scheduled sampling | Mix student's own predictions into training inputs (10%→30%) to fix exposure bias |
| Data mix | 60% OpenHermes chat + 40% QA (TriviaQA + GSM8K + custom factual pairs) |
| Teacher | Frozen 4-bit via bitsandbytes NF4 (8B), or BF16 (2B). Separate GPU for 8B |
| Skipped layers | Embedding + LM head kept in FP16 (critical for generation) |
| Optimizer | 8-bit AdamW via bitsandbytes (saves ~16 GB) |
| Eval | Repetition penalty 2.0, no_repeat_ngram_size=3, greedy + sampling fallback |

### Key Files

```
quantize/
├── run_v4.py         # CURRENT: v4.3 QAT — BitLinear (8B) / Progressive (2B), scheduled sampling
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
| Qwen3-8B | 4-bit teacher (GPU 0) + BitLinear student (GPU 1) | ~78 GB peak/GPU | 2x A100 80GB | Yes (v4.3) |
| Qwen3-8B | 4-bit teacher + BitLinear student (single GPU) | ~85 GB peak | 1x H100 96GB / 1x A100 160GB | Estimated |
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

**Key observations:**
- Loss still declining at step 225 (no plateau yet) — strong go signal
- Generation evolved from random gibberish to English word fragments in 200 steps
- CUDA illegal memory access at step 200 gen check (async error) — fixed with try/except + synchronize
- GPU stable at 33/78 GB throughout, no OOM
- `CUDA_LAUNCH_BLOCKING=1` needed for reliable gen checks (~30% slower but no crashes)

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

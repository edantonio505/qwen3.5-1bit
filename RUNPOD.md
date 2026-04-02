# RunPod Setup — Qwen 1-bit QAT

## Quick Start

```bash
# 1. Clone
git clone https://github.com/edantonio505/qwen3.5-1bit.git
cd qwen3.5-1bit

# 2. Install deps
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# 3. Run (auto-detects GPUs, VRAM, architecture)
PYTHONUNBUFFERED=1 python quantize/run_cloud.py --model Qwen/Qwen3-8B 2>&1 | tee run.log
```

## GPU Memory Requirements

Measured from actual training runs (teacher + student + optimizer + gradients + activations):

| Model | Teacher | Total VRAM | Minimum GPU | Cost |
|---|---|---|---|---|
| Qwen3.5-2B | BF16 (3.8 GB) | ~40 GB | 1x A40 48GB | ~$0.40/hr |
| Qwen3-8B | 4-bit (4 GB) | ~78 GB | 1x A100 80GB | ~$2/hr |
| Qwen3-8B | BF16 (16 GB) | ~100 GB | 2x L40S 48GB | ~$2/hr |

**Will NOT fit on 24GB GPUs** (RTX 3090/4090) — even the 2B model needs ~40 GB.

## What the Script Does

`run_cloud.py` auto-detects:
- Number of GPUs → splits teacher model across them
- Total VRAM → sets batch size accordingly
- Architecture (x86_64 vs aarch64) → warns if ARM

It then:
1. Loads teacher (4-bit via bitsandbytes) — frozen, provides soft targets
2. Loads student (BF16) — with 1-bit quantized linear layers
3. Trains with normalized logit MSE + cosine similarity + CE
4. Runs generation checks every 50 steps
5. Evaluates and saves to `quantize/runs/cloud/`

## What We've Learned (Critical Context)

### The core problem: generation collapse
Training loss converges perfectly (CE: 10→0.003, cosine: 0.8→0.3) but the
model outputs `\n\n` repeated 60 times during generation. This happens because:
- Teacher forcing during training gives the model correct input tokens
- During generation, the model must use its own (wrong) outputs
- With 1-bit weights, small errors cascade through 28 layers and collapse

### What we tried on DIGITS (128GB, GB10)

| Approach | Training | Generation |
|---|---|---|
| KL distillation | KL explodes to 3600+ | Empty |
| KL clamped + CE dominant | CE→0.003 | Empty |
| Normalized MSE + cosine | cos→0.35 | Empty |
| + Skip embed/LM head | Same | Empty (greedy), garbage (sampling) |
| + SubLN before every BitLinear | CE→1.3, cos→0.48 | Empty |

### Key findings
1. **KL divergence is WRONG for 1-bit** — explodes over 151k vocab. Use MSE + cosine instead.
2. **SubLN helps at init** but training pushes model back to \n\n attractor
3. **"Paris" moved from rank 134,940 to 18,016** in 375 steps — model IS learning, just not enough
4. **Need 10-100x more steps** — DIGITS is too slow (~4hrs for 1250 steps on 2B)
5. **Dynamic scales work** — learned scales had shape bugs with Qwen3.5 architecture

### What to try on RunPod (with 10x speed)

Priority order:
1. **Much more data + steps** — 50k examples, 5 epochs, 10k+ steps
2. **Ternary {-1, 0, +1}** — BitNet b1.58 approach, proven at scale. Modify STE1Bit to round to {-1,0,+1} instead of sign.
3. **Scheduled sampling** — during training, randomly use model's own generated tokens instead of teacher-forced tokens
4. **Hadamard rotation** — spread weight outliers before binarization

### Architecture notes for Qwen3.5
- `model.embed_tokens`: Embedding (NOT nn.Linear) — skip automatically
- `lm_head`: Linear — skip explicitly
- `in_proj_qkv`: Fused QKV projection, shape [6144, 2048] for 2B
- `in_proj_a`, `in_proj_b`: Small projections [16, 2048]
- `linear_attn`: Qwen3_5GatedDeltaNet (custom attention)
- Some layers have `sub_ln` (RMSNorm) added by our BitLinear wrapper

### PrismML Bonsai (for reference)
- True binary {-d, +d}, NOT ternary
- Q1_0_g128: 1 sign bit + FP16 scale per 128 weights = 1.125 BPW
- All layers quantized including embed + LM head
- Scale = mean(|w|) per group
- Bit balance exactly 50/50 — natively trained, not PTQ
- 70.5 avg benchmark at 1.15 GB (vs Qwen3-8B: 79.3 at 16.38 GB)
- Proprietary Caltech IP, no published research paper

## Files

```
quantize/
├── run.py            # Local training (v3: SubLN + dynamic scales)
├── run_cloud.py      # Cloud training (multi-GPU, 4-bit teacher, auto-detect)
├── quantize_lib.py   # Full library (progressive quant, Hadamard, learned scales)
├── auto_tune.py      # Hyperparameter search loop (8 configs)
├── evaluate.py       # Benchmark evaluation
├── export_gguf.py    # Export to Q1_0_g128 GGUF
└── train.py          # Standalone training script
```

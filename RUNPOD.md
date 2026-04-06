# RunPod Setup — Qwen 1-bit QAT

## Quick Start

```bash
# 1. Clone
git clone https://github.com/edantonio505/qwen3.5-1bit.git
cd qwen3.5-1bit

# 2. Install deps
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf

# 3. Run v9 (CURRENT BEST — proven architecture + 80% QA + 50k steps)
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 50000 \
    --max-examples 10000 --epochs 100 --lr 1e-4 \
    --qa-ratio 0.8 \
    --gen-check-interval 500 --eval-interval 2000 \
    --output-dir quantize/runs/v9-qwen3-8b \
    --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 --unlikelihood-weight 0 \
    2>&1 | tee run_v9.log

# First run (no GPTQ checkpoint — runs Phase 1 first, ~15 min):
# Remove --skip-gptq and --gptq-checkpoint flags
```

## GPU Memory Requirements

Measured from actual training runs (teacher + student + optimizer + gradients + activations):

| Model | Teacher | Quantizer | Total VRAM | Minimum GPU | Cost |
|---|---|---|---|---|---|
| Qwen3.5-2B | BF16 (3.8 GB) | ProgressiveQuantized | ~40 GB | 1x A40 48GB | ~$0.40/hr |
| Qwen3-8B | 4-bit NF4 (6.4 GB) | BitLinear | ~78 GB peak/GPU | 2x A100 80GB | ~$3/hr |
| Qwen3-8B | 4-bit NF4 | BitLinear (single GPU) | ~85 GB peak | 1x H100 96GB | ~$4/hr |

**Will NOT fit on 24GB GPUs** (RTX 3090/4090) — even the 2B model needs ~40 GB.

**Will NOT fit on 1x A100 80GB** for 8B — student alone peaks at ~78 GB, teacher needs ~6 GB more.

## What the Script Does

`run_v5.py` auto-detects:
- Number of GPUs → multi-GPU mode (teacher GPU 0, student GPU 1) or single GPU
- Total VRAM → sets batch size, seq length, grad accumulation
- Model size → uses BitLinear (8B+) or ProgressiveQuantizedLinear (2B)

It then runs a two-phase pipeline:

**Phase 1 — GPTQ Calibration (~10 min, single GPU):**
1. Loads FP16 model, runs 128 WikiText-2 calibration samples
2. Quantizes each layer with Hessian-weighted GPTQ + Hadamard rotation + sign-flip refinement
3. Saves calibrated checkpoint (optimal binary weights + group scales)
4. Frees GPU memory for Phase 2

**Phase 2 — QAT Fine-tuning (multi-GPU):**
1. Loads teacher (4-bit NF4) on GPU 0
2. Loads GPTQ-calibrated student on GPU 1 — BitLinear initialized from optimal binary weights
3. Trains with logit MSE (0.4) + cosine (0.2) + CE (0.4) + **hidden state MSE (0.1)**
4. Uses scheduled sampling (10%→30%)
5. Saves best checkpoint + final model

### Memory-critical details
- **BitLinear** (not ProgressiveQuantizedLinear) for 8B: saves signs as int8 (1 byte vs 2), no blending
- **8-bit AdamW** via bitsandbytes saves ~16 GB optimizer memory
- `del s_out` after extracting logits frees KV cache + hidden states
- `del s_logits, t_logits` before backward — only loss graph needed
- `zero_grad(set_to_none=True)` frees gradient tensors vs zeroing
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation

## What We've Learned (Critical Context)

### The core problem: generation collapse
Training loss converges perfectly (CE: 10→0.003, cosine: 0.8→0.3) but the
model outputs `\n\n` repeated 60 times during generation. This happens because:
- Teacher forcing during training gives the model correct input tokens
- During generation, the model must use its own (wrong) outputs
- With 1-bit weights, small errors cascade through 28+ layers and collapse
- **v4.3 mitigation:** Scheduled sampling (10%→30%) replaces some teacher tokens with student's own predictions

### Memory lessons learned the hard way

| Issue | Symptom | Fix |
|---|---|---|
| ProgressiveQuantizedLinear on 8B | OOM at 84+ GB (blending intermediates) | Switch to BitLinear (int8 signs, no blending) |
| Standard AdamW on 8B | OOM from ~32 GB optimizer states | Use 8-bit AdamW via bitsandbytes |
| transformers 5.5 + PyTorch 2.4 | `AttributeError: set_submodule` | Monkey-patch in run_v4.py |
| Logits not freed before backward | Peak memory too high | `del s_logits, t_logits` before `loss.backward()` |
| Gradients not freed after step | Memory stays high between steps | `zero_grad(set_to_none=True)` |
| KV cache fragments | OOM on generation checks | `torch.cuda.empty_cache()` before/after eval |
| CUDA async errors at gen checks | `illegal memory access` at `empty_cache()` | try/except + `torch.cuda.synchronize()` before gen, `CUDA_LAUNCH_BLOCKING=1` |
| 8-bit AdamW lazy init memory spike | 16 GB jump at step 2 (states created on first `opt.step()`) | Expected behavior — budget for it; without 8-bit would be 32 GB |
| On-policy + single-GPU student | OOM at 84 GB (2 forward passes per step) | Split student across both GPUs via `accelerate.dispatch_model()` |
| GPTQ binary values as weight init | Flat gradient landscape, loss worse than naive | Keep FP16 magnitudes, only flip signs to match GPTQ Hessian-optimal |
| Device mismatch with split model | `Expected all tensors on same device` | Explicit `.to(loss_device)` on all student outputs before loss computation |
| 35k examples insufficient for 1-bit | Loss plateaus, gen never improves | Raise to 300k+ examples, 20 epochs (OneBit used 13.5B tokens) |

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
2. **ProgressiveQuantizedLinear OOMs on 8B** — blending creates 3 intermediate tensors per layer. BitLinear with int8 signs fits.
3. **2x A100 80GB works** — teacher GPU 0 (6.4 GB), student GPU 1 (peaks 78 GB). Config: batch=1, seq=512, grad_accum=16.
4. **SubLN helps at init** but training pushes model back to \n\n attractor
5. **"Paris" moved from rank 134,940 to 18,016** in 375 steps — model IS learning, just not enough
6. **Need 10-100x more steps** — DIGITS is too slow (~4hrs for 1250 steps on 2B)
7. **Dynamic scales work** — learned scales had shape bugs with Qwen3.5 architecture
8. **8-bit AdamW is essential for 8B** — saves ~16 GB, difference between OOM and fitting

### v4.3 results (killed at step 300)
- Loss plateaued at 1.93 (delta only -0.12 over last 75 steps)
- Generation: random gibberish → English word fragments, but never correct answers
- **Root cause:** Naive sign(w) initialization leaves model in terrible basin
- Same pattern as every prior run: loss converges, generation collapses

### v5.0 (killed — GPTQ binary init hurt optimization)
- GPTQ Phase 1 OK (890k error, 0/8 eval — expected for 1-bit PTQ)
- Phase 2 loss started WORSE (8.89 vs v4.3's 8.62) — binary values gave flat gradients
- Root cause: initializing self.weight with GPTQ ±scale values, not smooth FP16

### v5.1 (OOM at step 25 — on-policy + single-GPU student = 84 GB)
### v5.3 (killed step ~120 — hyperparameters too conservative)
- Loss 5.79 at step 75 (slower than v4.3's 4.24)
- Gen at step 100: `, 01. a the is and in to` — new repetition attractor
- Unlikelihood 0.1 too weak, STE clip 1.0 too aggressive, on-policy ramp too slow

### v5.4 (killed step 75 — same trajectory as v5.3, hyperparameter tuning didn't help)

### v6 (killed step 1000 — 5-term loss causes degenerate attractors)
- SVID + 500k data. Loss 9.95→2.24 (best ever), BUT gen "Okayimport" attractor, 0/8 eval
- Root cause: multi-term loss creates conflicting gradients → degenerate generation modes
- 500k unique examples seen once < 30k examples with repetition

### v7 (killed — missing LayerNorm inside BitLinear)
Research agent audited OneBit's GitHub codebase, found 5 critical missing features.

### v8 run in progress (2026-04-05) — full OneBit architecture
- **FIX 1 (PRIMARY): LayerNorm(elementwise_affine=False) inside every SVIDBitLinear**
  → Prevents activation explosion during autoregressive generation
  → This was THE missing feature in all prior runs (v4.3 through v7)
- **FIX 2:** Tanh-STE: `grad * (1.001 - tanh(w)²)` (smooth gradient gate)
- **FIX 3:** NMF init for alpha/beta + weight=sign(W)*0.01 (max gradient flow)
- **FIX 4:** All-layer L2-normalized directional alignment (dominant loss term)
- **FIX 5:** LR 1e-4 (was 5e-6), beta2=0.98
- 30k examples × 50 epochs, SVID, simple loss, 10k steps
- **Architecture PROVEN** through step 4250: LayerNorm prevents generation collapse.
- First ever content words (step 600) and correct answer (step 2000, 2+2=4). Score 1/8.
- Score stuck at 1/8 from step 2000-4250. pkd plateaued at ~15.5. Killed.
- **Root cause: data insufficient.** "Paris" seen ~50 times (OneBit: 1000).

### v9 run in progress (2026-04-06) — data repetition fix
- Same architecture as v8 (proven). **80% QA ratio** (was 13%).
- 50k steps, 10k examples × 100 epochs. Each fact seen ~2,500 times.
- Tensorboard: `tensorboard --logdir quantize/runs/v9-qwen3-8b/tensorboard --bind_all`
- Checkpoints every 500 steps. Crash recovery with `--resume-from`.

### Architecture notes for Qwen3/Qwen3.5
- `model.embed_tokens`: Embedding (NOT nn.Linear) — skip automatically
- `lm_head`: Linear — skip explicitly
- Qwen3-8B has 252 quantizable linear layers + 1 skipped (lm_head)
- `in_proj_qkv`: Fused QKV projection, shape [6144, 2048] for 2B
- `in_proj_a`, `in_proj_b`: Small projections [16, 2048] (Qwen3.5 only)
- `linear_attn`: Qwen3_5GatedDeltaNet (Qwen3.5 custom attention, NOT in Qwen3)
- Qwen3-8B is a standard dense transformer — no hybrid attention

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
├── run_v5.py         # CURRENT: v8 — full OneBit architecture (LayerNorm + tanh-STE + NMF + all-layer alignment)
├── gptq_1bit.py      # GPTQ 1-bit PTQ: Hadamard, sign-flip refinement, layer-wise calibration
├── run_v4.py         # v4.3 — BitLinear QAT (loss converges but gen collapses)
├── run_cloud.py      # Cloud training (multi-GPU, 4-bit teacher, auto-detect, BitLinear)
├── run.py            # Legacy: local training with SubLN + dynamic scales
├── quantize_lib.py   # Full library (progressive quant, Hadamard, learned scales)
├── auto_tune.py      # Hyperparameter search loop (8 configs)
├── evaluate.py       # Benchmark evaluation
├── export_gguf.py    # Export to Q1_0_g128 GGUF
└── train.py          # Standalone training script
```

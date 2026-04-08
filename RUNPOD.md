# RunPod Setup — Qwen 1-bit QAT

> 🎉🎉 **v10 BREAKTHROUGH GROWING (2026-04-07):** The v8 architecture + 80% QA + 50k-step
> recipe on Qwen3-1.7B reached **6/8 = 75% accuracy** on the factual eval at step 10000.
> Six correct factual answers: Paris, 4, Pacific, Shakespeare, Au, 1945. pkd dropped from
> 65 → 8.0, blowing past v8's plateau of 15.5. **6x improvement over v8's previous best
> (1/8) on a smaller model.** Continued training added NEW facts (Pacific, 1945 were both
> wrong at step 6000, correct by step 10000). Best checkpoint at
> `quantize/runs/v10-qwen3-1.7b/best/`. See `CLAUDE.md` for the full v10 results and the
> qualitative phase-transition analysis.

> ⚠️ **MODEL CHOICE RULE:** Only dense Qwen3 models work — `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`,
> `Qwen3-8B`. **NEVER use anything from the Qwen3.5 family.** Qwen3.5 is a multimodal hybrid
> (`Qwen3_5ForConditionalGeneration`, vision tower, MTP head, 18/24 text-tower layers are
> Mamba-style `linear_attention`). Our 1-bit recipe only handles dense `nn.Linear` stacks.
> Verify before launch: model config must have `model_type: qwen3` and
> `architectures: ["Qwen3ForCausalLM"]`. The `flash-linear-attention` / `causal-conv1d` warning
> at load time is the giveaway you're on a hybrid model — abort.

> ⚠️ **DISK QUOTA TRAP:** RunPod's `/workspace` is a FUSE-mounted MooseFS network volume with
> a per-tenant quota (~50 GB on a typical pod). Each checkpoint is ~7 GB, so 6-7 of them fill
> the quota and the next checkpoint write gets **silently truncated** to a 1 GiB exact boundary
> mid-flight, killing the training process. Worse: any other file being written (e.g. an `Edit`
> to a python script) at the same moment can also be null-zeroed. Mitigation:
> `quantize/run_v5.py` now rotates checkpoints (deletes oldest before each save) to cap usage
> at ~14 GB. Verify before any large write operation: `du -sh /workspace`. Clean up if > 35 GB.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/edantonio505/qwen3.5-1bit.git
cd qwen3.5-1bit

# 2. Install deps
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf scikit-learn tensorboard

# 3. Run v10 (CURRENT — Qwen3-1.7B dense proof of concept, ~5x faster than 8B)
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-1.7B --use-4bit-teacher --max-steps 50000 \
    --max-examples 10000 --epochs 100 --lr 1e-4 \
    --qa-ratio 0.8 \
    --gen-check-interval 500 --eval-interval 2000 \
    --output-dir quantize/runs/v10-qwen3-1.7b \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 --unlikelihood-weight 0 \
    2>&1 | tee run_v10.log

# Hardware: 1.7B dense fits on 1x A40 48GB (~$0.40/hr) — ~85% cost savings vs 2x A100 80GB

# First run (no GPTQ checkpoint — runs Phase 1 first, ~3 min on dense 1.7B):
# Remove --skip-gptq and --gptq-checkpoint flags

# Sanity check after launch: GPTQ Phase 1 should print "28 layers, 7 linears each" with NO
# flash-linear-attention warning. If you see that warning, you launched on the wrong model.
```

## GPU Memory Requirements

Measured from actual training runs (teacher + student + optimizer + gradients + activations):

| Model | Teacher | Quantizer | Total VRAM | Minimum GPU | Cost |
|---|---|---|---|---|---|
| Qwen3-1.7B (dense) | 4-bit NF4 (1.3 GB) | SVIDBitLinear | ~20-30 GB peak | 1x A40 48GB | ~$0.40/hr |
| Qwen3-8B | 4-bit NF4 (6.4 GB) | SVIDBitLinear | ~78 GB peak/GPU | 2x A100 80GB | ~$3/hr |
| Qwen3-8B | 4-bit NF4 | SVIDBitLinear (single GPU) | ~85 GB peak | 1x H100 96GB | ~$4/hr |

**Will NOT fit on 24GB GPUs** (RTX 3090/4090).

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

### v9 (killed step 400) — same data plateau on 8B
- Faster initial pkd drop (3x faster to pkd~26) but plateaued at SAME level as v8
- More data alone doesn't break through. Either need more compute OR different approach.

### v10 first attempt (ABORTED 2026-04-07) — Wrong architecture
- Launched on Qwen3.5-2B without checking the model config first
- Killed during GPTQ Phase 1 once we noticed the `flash-linear-attention` warning
- Qwen3.5-2B is `Qwen3_5ForConditionalGeneration` — multimodal vision-LM with 18/24 text-tower
  layers as Mamba-style `linear_attention`. Our v8 recipe targets dense `nn.Linear` only;
  it has no LayerNorm fix for SSM recurrent state. Aborted dir:
  `quantize/runs/v10-qwen3.5-2b-ABORTED-hybrid-arch/`. See MODEL CHOICE RULE at top of file.

### v10 (BREAKTHROUGH GROWING 2026-04-07) — Qwen3-1.7B dense, 6/8 = 75% at step 10000
- Same v8 architecture on Qwen3-1.7B (~5x faster training than 8B)
- True dense twin of Qwen3-8B: same family (`Qwen3ForCausalLM`), same tokenizer (vocab 151936),
  28 dense layers, full attention every layer, no SSM/MoE/vision
- Survived a checkpoint-write disk-quota crash at step 3500; resumed cleanly from
  checkpoint-3000 with optimizer state preserved. Fixed via checkpoint rotation.
- **pkd: 65 → 8.0 by step 10000 (vs v8 plateau 15.5 on 8B).**
- **Eval progression: 0/8 (step 2000) → 4/8 = 50% (step 6000) → 6/8 = 75% (step 10000).**
  HITs at step 10000: Paris, "4", Pacific, Shakespeare, Au, 1945. MISSes: 144/12 (arithmetic),
  102°C (off by 2 from 100°C — calibration issue). **6x v8's previous best (1/8) on a smaller
  model.** Best checkpoint at `quantize/runs/v10-qwen3-1.7b/best/` (3.4 GB safetensors — DO
  NOT delete).
- **Phase transition observed:** between steps 6000 and 10000 the gen checks went from
  rambling-with-content (`"Madrid, Paris and gentlemen..."`) to single-word terminated
  (`"Paris"`). The model learned to STOP after the answer — a phase v8 never reached on 8B.
- Continued training adds NEW facts (Pacific, 1945 were wrong at step 6000, correct at 10000).
- Run continues to 50k steps to find the asymptote on 1.7B
- Fits on 1x A40 48GB (~$0.40/hr vs $3/hr for 2x A100 80GB), peak ~17/33 GB used
- Tensorboard: `tensorboard --logdir quantize/runs/v10-qwen3-1.7b/tensorboard --bind_all`

### Architecture notes for Qwen3 vs Qwen3.5
**Qwen3 (dense, supported):**
- `model.embed_tokens`: Embedding (NOT nn.Linear) — skip automatically
- `lm_head`: Linear — skip explicitly
- Qwen3-8B has 252 quantizable linear layers (28 layers × 9 minus skips)
- Qwen3-1.7B has 28 layers with 7 linears each (q/k/v/o + gate/up/down)
- Standard dense transformer, full attention every layer, no hybrid blocks

**Qwen3.5 (hybrid multimodal — DO NOT USE with this recipe):**
- `Qwen3_5ForConditionalGeneration` — multimodal vision-LM
- `vision_config` (vision tower with patch embeddings, image/video tokens)
- `mtp_num_hidden_layers: 1` — multi-token prediction head
- `layer_types`: text-tower has interleaved `linear_attention` (Mamba) and `full_attention`
  (e.g. 2B is 18/24 linear-attention, 6/24 full attention)
- `linear_conv_kernel_dim`, `mamba_ssm_dtype`, `linear_key_head_dim`, `linear_num_value_heads` —
  SSM dynamics that the LayerNorm-after-binary fix does not address
- `linear_attn`: Qwen3_5GatedDeltaNet (custom Mamba-style attention, NOT in Qwen3)
- Vocab 248320 (different tokenizer from Qwen3-8B's 151936)
- The `flash-linear-attention` / `causal-conv1d` warning at load time is the giveaway

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

# HANDOFF — How to Resume This Research on a New Server / New Claude Code Session

This document is the **canonical resume point** for the 1-bit Qwen quantization research
project. It's designed so that a fresh Claude Code session on any new machine, with no prior
context about this work, can pick up where we left off.

**Last updated:** 2026-04-07
**Last action:** Killed v10 on Qwen3-1.7B at step ~18900 after it peaked at 6/8 = 75% (step 10000) and regressed.
**Next action:** Implement early-stopping fix, then launch v11 on Qwen3-8B (the actual product).

---

## TL;DR — what is this project?

We're reproducing PrismML's Bonsai-8B (proprietary 1-bit Qwen3-8B at 70.5% avg benchmark)
using a fully open recipe. After 10 versions of training runs, we have:

- **A proven 1-bit QAT architecture** (v8 recipe): LayerNorm inside SVIDBitLinear, tanh-STE,
  NMF init, all-layer directional alignment, LR 1e-4, 80% QA data ratio.
- **A demonstrated result on Qwen3-1.7B**: 6/8 = 75% on an internal 8-question factual eval
  (Paris, 4, Pacific, Shakespeare, Au, 1945 — all correct at 1-bit). 6x v8's previous best.
- **A critical methodology fix needed**: 1-bit QAT runs overfit past their peak score while
  loss/pkd keeps dropping. Early stopping based on EVAL SCORE (not loss) is mandatory for
  the next run.

The next step is **v11**: launch the same recipe on Qwen3-8B with the early-stopping fix
applied. Expected to produce a Bonsai-class result.

---

## Current state (read this section first)

**v10 is DONE.** Process killed, GPU clean, no active training runs.

**Best result:** v10 step 10000 = **6/8 = 75%** on Qwen3-1.7B
- Checkpoint location (on this RunPod volume only): `quantize/runs/v10-qwen3-1.7b/best/`
- Files: `model.safetensors` (3.44 GB), `config.json`, `chat_template.jinja`,
  `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`
- **⚠️ This checkpoint is NOT in git. If this RunPod pod is deleted, the checkpoint is lost.**
  See "Checkpoint persistence" section below for backup options.

**Why v10 was killed:** The score regressed for 3 consecutive evals after step 10000:

| Step | pkd | Score | Notes |
|------|-----|-------|-------|
| 6000 | 9.2 | 4/8 = 50% | First breakthrough |
| **10000** | **8.0** | **6/8 = 75% PEAK** | Best checkpoint saved |
| 16000 | 7.5 | 5/8 = 62% | Lost "1945" → "August945" |
| 18000 | 7.4 | 4/8 = 50% | Lost "Pacific" → "Oceans" |

The model overfit to single-token answer style. France→"Paris" and 2+2→"4" became
the cleanest answers in project history (perfect single-token outputs), but the model
**traded breadth for depth** — losing multi-token answers (Pacific Ocean, 1945, 100°C
precision). **pkd kept dropping the entire time** the score was regressing, so loss-based
stopping criteria are useless for this regime.

---

## The settled architecture (DO NOT CHANGE)

After 10 versions of trial and error, the working 1-bit QAT recipe is locked in. **Do not
modify these unless you're prepared to debug from scratch.** All in `quantize/run_v5.py`.

1. **SVIDBitLinear with `nn.LayerNorm(out_features, elementwise_affine=False)` after every
   binary matmul.** This is THE primary fix. Without it, each 1-bit layer amplifies activation
   errors by O(√d), and by layer 20 the model has drifted from training distribution and
   generation collapses to high-frequency-token attractors. With it, generation stays in
   distribution.
2. **Tanh-STE**: gradient gate `grad * (1.001 - tanh(w)²)`. Smooth gate, plastic near 0,
   frozen far from 0. Better than vanilla STE or hard-clipped STE.
3. **NMF init for alpha/beta** (sklearn rank-1 NMF on |W|) + **weight = sign(W) * 0.01**
   (small magnitude for max gradient flow under tanh-STE).
4. **All-layer L2-normalized directional alignment as the dominant loss** (`pkd_loss`). KD
   logit loss is scaled down 100x.
5. **LR 1e-4, beta2=0.98, 8-bit AdamW.** Earlier runs used 5e-6 (80x too low) and the sign
   landscape was frozen.
6. **80% QA data ratio** (`--qa-ratio 0.8`). Without this, QA examples are seen too few times
   for fact memorization.
7. **True binary {-1, +1}** — never ternary {-1, 0, +1}.

The launch command for v10 (which produced the 6/8 result) is in the next section.

---

## Required code changes BEFORE launching v11

These are NOT yet in the codebase. They must be added before v11 launches, or v11 will
repeat v10's overfitting cliff.

### Change 1: Early stopping by eval score (MANDATORY)

**Why:** v10 demonstrated that loss/pkd keeps improving while eval score regresses past the
peak. The "save best" logic in `run_v5.py` correctly preserves the peak checkpoint, but
doesn't stop the run from continuing to overfit and burn compute.

**What to add to `quantize/run_v5.py`:**

In the eval block (currently around line 1314-1340), add a counter `evals_since_best`:

```python
# Add early-stopping state (initialize before the training loop)
evals_since_best = 0
early_stop_patience = args.early_stop_patience  # CLI flag, default 3

# In the eval block, after computing `score`:
if score > best_score:
    best_score = score
    best_step = step
    evals_since_best = 0
    # ... existing best-save logic ...
else:
    evals_since_best += 1
    print(f"  No improvement ({evals_since_best}/{early_stop_patience} evals since best={best_score:.0f}% at step {best_step})")
    if evals_since_best >= early_stop_patience:
        print(f"  Early stopping: {early_stop_patience} consecutive evals without improvement.")
        print(f"  Best score: {best_score:.0f}% at step {best_step}")
        break  # exit the training loop
```

Add the CLI flag in the argument parser:
```python
parser.add_argument("--early-stop-patience", type=int, default=3,
                    help="Stop if N consecutive evals fail to beat the best score (default 3)")
```

### Change 2: Expand the eval set (STRONGLY RECOMMENDED)

**Why:** v10's 8-question eval is too noisy to reliably detect the peak (a single question
flipping = 12.5% score change). With a 50-100 question eval, the signal would be much more
stable and the early-stopping logic would have less false-positive noise.

**Where:** `run_eval()` in `quantize/run_v5.py` (search for the QUESTIONS list).

**What to add:** Hand-pick 50-100 questions from MMLU-mini, TriviaQA-easy, GSM8K-elementary.
Or wire in `lm-evaluation-harness` as a subprocess call. The current 8-question eval can stay
for backward compatibility, but the early-stopping decision should use the larger set.

### Change 3: Verify disk quota on new pod

**Why:** RunPod's `/workspace` is a FUSE-mounted MooseFS network volume with a per-tenant
quota that varies by pod. v10 crashed at step 3500 because the quota was hit during a
checkpoint write — silently truncating the file to a 1 GiB exact boundary and killing the
process. Different pods have different quotas; never trust `df -h`.

**What to run before launching v11:**
```bash
cd /workspace
dd if=/dev/zero of=test.bin bs=1M count=8192 conv=fsync
# Should succeed and write ~8 GB. If you get "Disk quota exceeded", you have a tight quota
# and need to either find a different pod or be very careful with checkpoint rotation.
rm test.bin
```

### Change 4: Verify checkpoint rotation logic is in place

**Why:** v10's crash recovery added a rotation block to `quantize/run_v5.py:1294-1308` (or
thereabouts after line shifts). Without rotation, checkpoints accumulate at ~7 GB each and
eventually fill the quota. Verify it's still there:

```bash
grep -A 12 "Periodic checkpoint" quantize/run_v5.py
```

You should see code that does `shutil.rmtree(oldest_path)` before each save. If it's missing
(e.g., if someone reverted it), re-add it. The block requires `import shutil` at the top.

---

## Launch command for v11 (Qwen3-8B return-trip)

**After all four code changes above are made and verified**, launch v11:

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
  python3 quantize/run_v5.py \
    --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 50000 \
    --max-examples 10000 --epochs 100 --lr 1e-4 --qa-ratio 0.8 \
    --gen-check-interval 500 --eval-interval 2000 \
    --output-dir quantize/runs/v11-qwen3-8b \
    --skip-gptq --gptq-checkpoint quantize/runs/v5-qwen3-8b/gptq_checkpoint \
    --use-svid --simple-loss \
    --on-policy-fraction 0.15 --on-policy-len 32 --ste-clip 0 --unlikelihood-weight 0 \
    --early-stop-patience 3 \
    > run_v11.log 2>&1 &
```

**Notes:**
- `--early-stop-patience 3` only works after Change 1 above is implemented.
- `--skip-gptq` reuses the 8B GPTQ Phase 1 checkpoint that was saved during a prior 8B run
  (`quantize/runs/v5-qwen3-8b/gptq_checkpoint/`). Saves ~10 minutes. **This checkpoint is
  also on this pod only and will be lost if the pod is deleted.** If you're on a fresh
  machine, remove `--skip-gptq` and the `--gptq-checkpoint` flag — Phase 1 will run from
  scratch (~10 min on 8B).
- Hardware: **2x A100 80GB** required for 8B. The A40 48GB v10 used is NOT sufficient for
  8B QAT (student peaks at 78 GB on a single GPU, requires split across 2 GPUs).

---

## Hardware required for each model size

| Model | Hardware | Approx cost | Verified |
|-------|---|---|---|
| Qwen3-1.7B (dense, v10) | 1x A40 48GB | ~$0.40/hr | YES — peak ~17/33 GB |
| Qwen3-4B (dense) | 1x A100 80GB | ~$1.50/hr | Estimated |
| Qwen3-8B (dense, v11 target) | **2x A100 80GB** | ~$3/hr | YES (v5.3) — peak ~50-60 GB/GPU |

For v11, budget ~$3/hr × 5-9 days = **$360-650** total. Early stopping (Change 1 above)
should cut this significantly because v10 peaked at step 10000 (out of 50000 planned), so
v11 may only run a fraction of its full budget.

---

## ⚠️ MODEL CHOICE RULE — never violate

**Only use dense Qwen3 models:** `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`.

**NEVER use anything from the Qwen3.5 family.** Qwen3.5-2B (and the rest of Qwen3.5) is a
**multimodal vision-language hybrid** with a vision tower, MTP head, and 18/24 layers as
Mamba-style `linear_attention` (SSM). Our v8 recipe is built for dense `nn.Linear` stacks
and does NOT transfer to SSM/multimodal layers.

**Verification before any launch:**
```bash
# Check the model config
python3 -c "import json; c = json.load(open('/path/to/config.json')); print('model_type:', c.get('model_type')); print('arch:', c.get('architectures'))"
```

You MUST see:
- `model_type: qwen3` (NOT `qwen3_5`)
- `architectures: ["Qwen3ForCausalLM"]` (NOT `Qwen3_5ForConditionalGeneration` or anything
  with "Vision", "MoE", "Mamba", "Hybrid")

If the load-time logs show a `flash-linear-attention` / `causal-conv1d` warning, **abort
immediately** — that's the giveaway you're on a hybrid model.

**Why this matters:** v10's first attempt was launched on Qwen3.5-2B as a "smaller proof of
concept" and immediately killed when we noticed the architecture mismatch. The aborted dir
is at `quantize/runs/v10-qwen3.5-2b-ABORTED-hybrid-arch/` as a reminder.

---

## Setup on a fresh server (RunPod or any Linux + CUDA)

```bash
# 1. Clone the repo
git clone https://github.com/edantonio505/qwen3.5-1bit.git
cd qwen3.5-1bit

# 2. Install dependencies
pip install torch transformers datasets accelerate bitsandbytes sentencepiece protobuf scikit-learn tensorboard

# 3. Read the critical context (this file plus the others)
cat HANDOFF.md     # this file
cat CLAUDE.md      # detailed project state and v10 result tables
cat README.md      # public-facing summary
cat RUNPOD.md      # RunPod-specific gotchas (disk quota trap, etc.)

# 4. Verify hardware
nvidia-smi
# For 1.7B: 1x A40 48GB or better
# For 8B (v11): 2x A100 80GB or equivalent

# 5. Verify disk quota (CRITICAL — see Change 3 above)
cd /workspace  # or wherever your training output dir lives
dd if=/dev/zero of=test.bin bs=1M count=8192 conv=fsync && rm test.bin
# If this fails with "Disk quota exceeded", investigate before proceeding

# 6. Implement the four required changes (early stopping, eval expansion, quota check, rotation)
# See "Required code changes BEFORE launching v11" section above

# 7. Verify the model is the right architecture (NOT Qwen3.5)
# See "MODEL CHOICE RULE" section above

# 8. Launch v11 (the 8B return-trip)
# See "Launch command for v11" section above
```

---

## All key file paths in the repo

```
quantize/
├── run_v5.py             # Main training script (despite name, contains the v8 architecture)
├── gptq_1bit.py          # GPTQ Phase 1 calibration
├── quantize_lib.py       # Shared library (Hadamard, STE variants, etc.)
├── diagnose.py           # Logit ranking diagnostic — run on a saved checkpoint
├── export_gguf.py        # Export to Q1_0_g128 GGUF format (after v11 succeeds)
├── evaluate.py           # Benchmark evaluation
└── runs/                 # Output dirs (NOT in git — local to each pod)
    ├── v10-qwen3-1.7b/
    │   ├── best/                    # ⭐ THE 6/8 = 75% checkpoint
    │   ├── checkpoint-XXXX/         # Periodic checkpoints (rotated, max 2)
    │   ├── gptq_checkpoint/         # 1.7B GPTQ Phase 1 (skip with --skip-gptq)
    │   └── tensorboard/
    ├── v8-qwen3-8b/best/            # v8's best (1/8 = 12.5%)
    └── v5-qwen3-8b/gptq_checkpoint/ # 8B GPTQ Phase 1 (reuse for v11)

CLAUDE.md      # Detailed project state with v10 result tables
README.md      # Public-facing summary
RUNPOD.md      # RunPod-specific gotchas
HANDOFF.md     # This file — resume point for new servers
```

The `quantize/runs/` directory is gitignored. **Checkpoints do not travel with the git repo.**
See "Checkpoint persistence" below.

---

## Run history (one-line per run, full details in CLAUDE.md)

| Run | Model | Result | Notes |
|-----|---|---|---|
| v4.3 | 8B | 0/8, killed step 300 | Naive sign(w) init, gen collapsed |
| v5.0 | 8B | 0/8, killed step 3 | GPTQ binary init gave flat gradients |
| v5.1 | 8B | OOM step 25 | On-policy + single-GPU = 84 GB |
| v5.3 | 8B | 0/8, killed step 120 | Hyperparameters too conservative |
| v5.4 | 8B | 0/8, killed step 75 | Same as v5.3 |
| v6 | 8B | 0/8 ("Okayimport"), killed step 1000 | 5-term loss → conflicting gradients |
| v7 | 8B | 0/8, killed | Missing LayerNorm in BitLinear |
| **v8** | 8B | **1/8 = 12.5%, killed step 4250** | **First architecture breakthrough**: LayerNorm + tanh-STE + NMF + all-layer alignment |
| v9 | 8B | killed step 400 | Same plateau as v8, 80% QA didn't help in time |
| v10 first try | Qwen3.5-2B | ABORTED | Wrong architecture (multimodal hybrid) |
| **v10** | **1.7B** | **6/8 = 75% PEAK at step 10000** | **All-time best.** Killed at step ~18900 due to overfitting regression |
| **v11** (planned) | **8B** | **TBD** | The actual product. Requires early-stopping fix. |

---

## Critical lessons learned (do not relearn the hard way)

1. **Loss/pkd is NOT a reliable proxy for benchmark score in 1-bit QAT.** v10 demonstrated
   that pkd kept dropping (8.0 → 7.4) while score regressed (6/8 → 4/8) over 8000 steps.
   Always use score-based early stopping. See "Required code changes" Change 1.

2. **The model overfits to terse answer style after the score peak.** Continued training
   produces cleaner single-token answers (Paris, 4) but loses multi-token answers (Pacific
   Ocean, 1945). The same trend that produces the breakthrough produces the regression —
   it's one continuous process.

3. **80% QA data ratio is essential.** v8 used 13% and plateaued at 1/8. v10 used 80% and
   reached 6/8. Without enough QA exposure, factual recall doesn't develop.

4. **LayerNorm inside BitLinear is the primary architectural fix.** Without it, generation
   collapses to high-frequency-token attractors during autoregressive decoding even when
   training loss looks fine.

5. **Tanh-STE > vanilla STE > clipped STE.** Smooth gradient gate is essential.

6. **NMF init + weight = sign(W) * 0.01** (small magnitude). With tanh-STE, small init
   gives 100% gradient flow at start; large init (e.g., sign(W) * 1.0) gives only 42%.

7. **LR 1e-4 (not 5e-6).** Earlier runs used 80x too low; sign landscape was frozen.

8. **NEVER use Qwen3.5.** It's multimodal hybrid. Use only dense `Qwen3-{0.6B,1.7B,4B,8B}`.
   See MODEL CHOICE RULE above.

9. **Disk quota traps on RunPod's MooseFS volume.** Per-tenant quotas truncate writes
   silently to 1 GiB exact boundaries. Verify with `dd` test before launching, and ensure
   checkpoint rotation is enabled in `run_v5.py`. See `RUNPOD.md` for the full story.

10. **Continued training CAN add new facts to a 1-bit model.** v10 went from 0/8 → 4/8 →
    6/8 with new facts (Pacific, 1945) appearing at step 10000 that weren't there at
    step 6000. This is a genuinely positive finding that contradicts naive intuitions
    about parametric capacity at 1-bit.

---

## Checkpoint persistence (IMPORTANT)

**The v10 best checkpoint (6/8 = 75%) lives at `quantize/runs/v10-qwen3-1.7b/best/` on this
specific RunPod pod. It is NOT in git. If this pod is deleted, the checkpoint is lost.**

**Backup options before tearing down this pod:**

1. **Upload to Hugging Face Hub** (recommended for sharing):
   ```bash
   pip install huggingface_hub
   huggingface-cli login  # paste your HF token
   huggingface-cli upload edantonio505/qwen3-1.7b-1bit-v10 \
       quantize/runs/v10-qwen3-1.7b/best/ \
       --repo-type=model --commit-message="v10 6/8 = 75% best checkpoint"
   ```

2. **Copy to RunPod persistent storage** (if you have a separate volume):
   ```bash
   cp -r quantize/runs/v10-qwen3-1.7b/best/ /persistent_volume/v10-best/
   ```

3. **Download to your local machine via `runpodctl` or `scp`**:
   ```bash
   # From your laptop:
   scp -r runpod-pod:/workspace/qwen3.5-1bit/quantize/runs/v10-qwen3-1.7b/best/ ./
   ```

The same applies to the **8B GPTQ checkpoint** at `quantize/runs/v5-qwen3-8b/gptq_checkpoint/`
(if you want to skip Phase 1 on v11) and the **1.7B GPTQ checkpoint** at
`quantize/runs/v10-qwen3-1.7b/gptq_checkpoint/`. These are also pod-only.

**On a fresh server, if these checkpoints aren't available**, you can:
- For v11: just remove `--skip-gptq` and `--gptq-checkpoint` flags (Phase 1 runs from scratch in ~10 min for 8B).
- For comparing against v10: you'd have to retrain v10 from scratch (~6-10 hours on A40 to
  reach the step 10000 peak). The recipe is documented in this file.

---

## Plan for Phase 3 (after v11 produces a checkpoint)

Once v11 produces a high-scoring 8B checkpoint, the next steps are:

1. **Run real benchmarks** via `lm-evaluation-harness`:
   ```bash
   pip install lm-eval
   lm_eval --model hf \
     --model_args pretrained=quantize/runs/v11-qwen3-8b/best \
     --tasks mmlu,arc_challenge,gsm8k,truthfulqa,hellaswag \
     --batch_size 1
   ```
   This is what PrismML reports for Bonsai (70.5% average). Without these, we can't claim parity.

2. **Export to Q1_0_g128 GGUF** via `quantize/export_gguf.py`. The PrismML inference path
   in `scripts/run_llama.sh` consumes this format — once exported, our model can run through
   their llama.cpp fork.

3. **Compare side-by-side with PrismML's Bonsai-8B:**
   - Same 5 benchmarks
   - Same prompts
   - Same hardware
   - File size, tokens/sec, accuracy

4. **(Optional) Write up findings** as a technical report or blog post. The contribution is
   a fully open recipe matching proprietary work — currently no public method exists.

---

## When in doubt, references in priority order

1. **This file (`HANDOFF.md`)** — start here on a new server
2. **`CLAUDE.md`** — detailed project state, full v10 eval tables, run history
3. **`README.md`** — public-facing summary
4. **`RUNPOD.md`** — RunPod-specific gotchas (disk quota, model warnings)
5. **`quantize/run_v5.py`** — the actual training code (the v8 architecture is in this file
   despite the name "v5")
6. **OneBit codebase** (https://github.com/xuyuzhuang11/OneBit) — the upstream source for
   all the architectural fixes (LayerNorm, tanh-STE, NMF, all-layer alignment). The OneBit
   *paper* doesn't describe these — only the GitHub code does.

---

## Last words

The architecture is settled. v10 proved it works. The remaining work on v11 is pure
engineering: add early stopping, expand the eval set, launch on bigger hardware, validate
against real benchmarks. There are no more architectural unknowns.

If you see anything in this document that seems wrong, **trust the code and the checkpoints
over this doc**. This is a snapshot in time and may drift from reality. Always verify with:
- `git log --oneline | head -20` for recent commits
- `ls quantize/runs/` for what training artifacts actually exist
- `grep -n "Score:" run_v*.log` for the actual eval scores
- `python3 -c "import safetensors; ..."` to inspect a checkpoint

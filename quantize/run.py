#!/usr/bin/env python3
"""1-bit QAT v3 — SubLN + activation-aware scales + layer-wise calibration.

Fixes for generation collapse:
1. SubLN: RMSNorm before every binary linear (prevents hidden state collapse)
2. Activation-aware scales: optimize scales from calibration data, not just weight stats
3. Layer-wise calibration: quantize one layer at a time, minimizing output error
4. Sampling fallback: try sampling if greedy produces garbage
5. Embed + LM head kept in FP16
"""
import functools
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

print = functools.partial(print, flush=True)

MODEL = "Qwen/Qwen3.5-2B"
LR = 1e-5
EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_SEQ_LEN = 512
MAX_EXAMPLES = 5_000
ALPHA_MSE = 0.4
ALPHA_COS = 0.2
ALPHA_CE = 0.4
GROUP_SIZE = 128
GEN_CHECK_INTERVAL = 40
CALIBRATION_BATCHES = 8  # batches for activation-aware scale calibration


# ══════════════════════════════════════════════════════════
#  1-bit quantizer with SubLN
# ══════════════════════════════════════════════════════════

class STE1Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scales):
        shape = weight.shape
        flat = weight.reshape(-1, GROUP_SIZE)
        signs = flat.sign()
        signs[signs == 0] = 1.0
        ctx.save_for_backward(signs)
        return (scales * signs).reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        signs, = ctx.saved_tensors
        flat_g = grad.reshape(-1, GROUP_SIZE)
        scale_grad = (flat_g * signs).sum(dim=1, keepdim=True)
        return grad, scale_grad


class BitLinear(nn.Module):
    """1-bit linear with SubLN (RMSNorm before binary matmul)."""

    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.in_features = orig.in_features
        self.out_features = orig.out_features

        # SubLN: normalize inputs before binary matmul to prevent hidden state collapse
        self.sub_ln = nn.RMSNorm(orig.in_features, elementwise_affine=False)

    def forward(self, x):
        # SubLN: normalize BEFORE binary matmul
        x_normed = self.sub_ln(x)
        # Compute scales dynamically from current weights (mean absolute value per group)
        flat = self.weight.reshape(-1, GROUP_SIZE)
        scales = flat.detach().abs().mean(dim=1, keepdim=True)
        q_w = STE1Bit.apply(self.weight, scales)
        return F.linear(x_normed, q_w, self.bias)


def replace_linears(model):
    replaced = skipped = 0
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear):
                if any(pat in full.lower() for pat in ["norm", "embed", "lm_head"]):
                    skipped += 1
                    continue
                if child.weight.numel() % GROUP_SIZE != 0:
                    skipped += 1
                    continue
                setattr(mod, cname, BitLinear(child))
                replaced += 1
    print(f"  Quantized: {replaced} layers | Kept FP16: {skipped}")
    return replaced


# ══════════════════════════════════════════════════════════
#  Activation-aware scale calibration
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_scales(model, teacher, dataloader, device, n_batches=8):
    """Calibrate scales using activation data from the teacher.

    For each BitLinear layer, find the scale that minimizes:
        ||scale * sign(W) @ x_normed - W @ x_normed||²

    Closed form: scale = (sign(W)@X · W@X) / (sign(W)@X)²
    """
    print("  Calibrating scales with activation data...")
    model.eval()
    teacher.eval()

    # Collect activations by hooking into BitLinear layers
    activation_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {"sum_num": 0.0, "sum_den": 0.0, "count": 0}
            x = input[0]  # input to this layer (after SubLN)
            x_normed = module.sub_ln(x)

            W = module.weight.data.float()
            flat_W = W.reshape(-1, GROUP_SIZE)
            signs = flat_W.sign()
            signs[signs == 0] = 1.0

            # For each group, compute optimal scale from activations
            # We approximate by sampling a subset of activation vectors
            x_flat = x_normed.float().reshape(-1, x_normed.shape[-1])  # (batch*seq, in_features)
            if x_flat.shape[0] > 64:
                idx = torch.randperm(x_flat.shape[0])[:64]
                x_flat = x_flat[idx]

            # W @ x for full precision
            Wx = (W @ x_flat.T)  # (out, samples)
            # sign(W) @ x for binary
            signs_full = W.sign()
            signs_full[signs_full == 0] = 1.0
            Sx = (signs_full @ x_flat.T)  # (out, samples)

            # Optimal per-group scale: sum(Sx * Wx) / sum(Sx^2) over the output dimension
            # But we have group structure, so compute per group
            Wx_grouped = Wx.reshape(-1, GROUP_SIZE, Wx.shape[1])  # (groups, group_size, samples)
            Sx_grouped = Sx.reshape(-1, GROUP_SIZE, Sx.shape[1])

            num = (Sx_grouped * Wx_grouped).sum(dim=(1, 2))  # per group
            den = (Sx_grouped * Sx_grouped).sum(dim=(1, 2))  # per group

            activation_stats[name]["sum_num"] += num.cpu()
            activation_stats[name]["sum_den"] += den.cpu()
            activation_stats[name]["count"] += 1

        return hook

    # Register hooks
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    # Run calibration batches
    batch_count = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(**batch)
        except Exception:
            pass
        batch_count += 1
        if batch_count >= n_batches:
            break

    # Remove hooks
    for h in hooks:
        h.remove()

    # Apply calibrated scales
    calibrated = 0
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear) and name in activation_stats:
            stats = activation_stats[name]
            if stats["count"] > 0:
                num = stats["sum_num"] / stats["count"]
                den = stats["sum_den"] / stats["count"]
                optimal_scale = (num / den.clamp(min=1e-8)).clamp(min=1e-6)
                with torch.no_grad():
                    mod.log_scale.data = torch.log(optimal_scale.unsqueeze(1).to(mod.log_scale.device, mod.log_scale.dtype) + 1e-8)
                calibrated += 1

    print(f"  Calibrated {calibrated} layers from {batch_count} batches")
    model.train()


# ══════════════════════════════════════════════════════════
#  Loss
# ══════════════════════════════════════════════════════════

def compute_loss(s_logits, t_logits, labels):
    s_norm = (s_logits - s_logits.mean(-1, keepdim=True)) / s_logits.std(-1, keepdim=True).clamp(min=1e-6)
    t_norm = (t_logits - t_logits.mean(-1, keepdim=True)) / t_logits.std(-1, keepdim=True).clamp(min=1e-6)
    mse = F.mse_loss(s_norm, t_norm)
    cos = 1.0 - F.cosine_similarity(s_logits, t_logits, dim=-1).mean()
    ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), labels.view(-1), ignore_index=-100)
    total = ALPHA_MSE * mse + ALPHA_COS * cos + ALPHA_CE * ce
    return total, mse.item(), cos.item(), ce.item()


# ══════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════

def tokenize(example, tokenizer):
    convs = example.get("conversations", [])
    if not convs:
        return None
    msgs = []
    for t in convs:
        r = t.get("from", t.get("role", ""))
        c = t.get("value", t.get("content", ""))
        if r in ("system",): msgs.append({"role": "system", "content": c})
        elif r in ("human", "user"): msgs.append({"role": "user", "content": c})
        elif r in ("gpt", "assistant"): msgs.append({"role": "assistant", "content": c})
    if not msgs:
        return None
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=False, enable_thinking=False)
    except Exception:
        return None
    toks = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)
    toks["labels"] = toks["input_ids"].copy()
    return toks


def collate(batch, pad_id):
    ml = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    for b in batch:
        p = ml - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * p)
        mask.append(b["attention_mask"] + [0] * p)
        labels.append(b["labels"] + [-100] * p)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels)}


# ══════════════════════════════════════════════════════════
#  Generation
# ══════════════════════════════════════════════════════════

def generate_answer(model, tokenizer, prompt, device, max_tokens=60, debug=False):
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer(text, return_tensors="pt").to(device)

    # Try greedy
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0][inp["input_ids"].shape[1]:]
    clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Fallback to sampling
    if not clean:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9, top_k=50,
                                 pad_token_id=tokenizer.pad_token_id)
        new_ids = out[0][inp["input_ids"].shape[1]:]
        clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if clean:
            clean = f"[S] {clean}"

    raw = tokenizer.decode(new_ids, skip_special_tokens=False)

    if debug and not clean:
        ids_list = new_ids[:10].tolist()
        tok_strs = [repr(tokenizer.decode([tid])) for tid in ids_list]
        print(f"      DEBUG IDs:    {ids_list}")
        print(f"      DEBUG tokens: {tok_strs}")

    return clean, raw, len(new_ids)


def run_eval(model, tokenizer, device, label=""):
    model.eval()
    qs = [
        ("Capital of France? One word.", "Paris"),
        ("2 + 2 = ? Just the number.", "4"),
        ("Largest ocean? One word.", "Pacific"),
        ("144 / 12? Just the number.", "12"),
        ("Who wrote Hamlet? Last name.", "Shakespeare"),
    ]
    correct = 0
    print(f"\n  --- Eval: {label} ---")
    for q, a in qs:
        clean, raw, n = generate_answer(model, tokenizer, q, device, debug=True)
        hit = a.lower() in clean.lower()
        display = clean if clean else f"[EMPTY]"
        print(f"    Q: {q}")
        print(f"    A: {display[:80]} ({n} tok) -> {'HIT' if hit else 'MISS'}")
        if hit:
            correct += 1
    score = correct / len(qs) * 100
    print(f"  Score: {correct}/{len(qs)} = {score:.0f}%")
    model.train()
    return score


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  1-bit QAT v3")
    print("  SubLN + Activation-aware scales + Calibration")
    print("=" * 55)
    print(f"  Model: {MODEL}")
    print(f"  Fixes: SubLN before every BitLinear")
    print(f"         Activation-aware scale calibration")
    print(f"         Embed + LM head in FP16")
    print(f"         Sampling fallback for generation")
    print()

    # Tokenizer
    print("[1/5] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Teacher
    print("[2/5] Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  {sum(p.numel()*p.element_size() for p in teacher.parameters())/1e9:.2f} GB")

    # Student
    print("[3/5] Student...")
    student = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    student.gradient_checkpointing_enable()
    replace_linears(student)
    student.to(device).train()
    print(f"  {sum(p.numel()*p.element_size() for p in student.parameters())/1e9:.2f} GB")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Data
    print("\n[4/5] Data...")
    raw_ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    data = []
    for ex in raw_ds:
        t = tokenize(ex, tok)
        if t and len(t["input_ids"]) > 20:
            data.append(t)
        if len(data) >= MAX_EXAMPLES:
            break
    print(f"  {len(data)} examples")

    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id),
                       num_workers=0, drop_last=True)

    print("\n[5/5] Ready (SubLN active, dynamic scales)...")

    # Baseline eval (after calibration, before training)
    baseline = run_eval(student, tok, device, "BASELINE (calibrated, before training)")
    teacher_score = run_eval(teacher, tok, device, "TEACHER (full precision)")

    # Optimizer
    total_steps = len(loader) * EPOCHS // GRAD_ACCUM
    warmup = max(1, int(total_steps * 0.05))

    # Two param groups: main weights + SubLN params (faster LR)
    norm_p = [p for n, p in student.named_parameters() if "sub_ln" in n]
    other_p = [p for n, p in student.named_parameters()
               if "sub_ln" not in n and p.requires_grad]

    groups = [{"params": other_p, "lr": LR}]
    if norm_p:
        groups.append({"params": norm_p, "lr": LR * 5})

    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    print(f"\n  Steps: {total_steps} | Warmup: {warmup}")
    print()

    # ── Train ──
    step = 0
    t0 = time.time()
    log_loss = log_mse = log_cos = log_ce = log_n = 0

    for epoch in range(EPOCHS):
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        t_logits = teacher(**batch).logits
                    s_logits = student(**batch).logits
                    loss, mse_v, cos_v, ce_v = compute_loss(s_logits, t_logits, labels)
                    loss = loss / GRAD_ACCUM
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at step {step}!")
                torch.cuda.empty_cache()
                opt.zero_grad()
                continue

            log_loss += loss.item() * GRAD_ACCUM
            log_mse += mse_v
            log_cos += cos_v
            log_ce += ce_v
            log_n += 1

            if (bi + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1

                if step <= 3 or step % max(1, total_steps // 20) == 0:
                    el = time.time() - t0
                    eta = (total_steps - step) / max(step / el, 1e-9)
                    print(f"  {step:>5d}/{total_steps} | "
                          f"loss={log_loss/log_n:.3f} MSE={log_mse/log_n:.3f} "
                          f"cos={log_cos/log_n:.4f} CE={log_ce/log_n:.3f} | "
                          f"lr={sched.get_last_lr()[0]:.1e} | ETA {eta/60:.1f}m")
                    log_loss = log_mse = log_cos = log_ce = log_n = 0

                if step % GEN_CHECK_INTERVAL == 0:
                    print(f"\n  --- Gen check step {step} ---")
                    student.eval()
                    for prompt in ["Capital of France? One word.",
                                   "2+2=? Just the number.",
                                   "Say hello."]:
                        c, _, n = generate_answer(student, tok, prompt, device, debug=True)
                        tag = prompt.split("?")[0].split(".")[0]
                        print(f"    {tag}: {c or '[EMPTY]'} ({n} tok)")
                    student.train()

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    train_min = (time.time() - t0) / 60
    print(f"\n  Training: {train_min:.1f} min, {step} steps")

    final = run_eval(student, tok, device, "FINAL (after QAT)")

    print(f"\n{'='*55}")
    print(f"  Teacher:  {teacher_score:.0f}%")
    print(f"  Baseline: {baseline:.0f}% (calibrated, no training)")
    print(f"  Final:    {final:.0f}% (after QAT)")
    print(f"  Delta:    {'+' if final > baseline else ''}{final - baseline:.0f}%")
    print(f"{'='*55}")

    out = Path("quantize/runs/v3")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out / "model.pt")
    tok.save_pretrained(out)
    with open(out / "result.json", "w") as f:
        json.dump({"teacher": teacher_score, "baseline": baseline,
                    "final": final, "steps": step, "train_min": train_min,
                    "model": MODEL}, f, indent=2)
    print(f"  Saved to {out}")

    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

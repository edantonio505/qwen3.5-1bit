#!/usr/bin/env python3
"""1-bit QAT with working distillation. v2.

Key changes from v1:
- Replaced KL divergence with normalized logit MSE (doesn't explode)
- Added cosine similarity loss on logits (scale-invariant)
- 10x more data to prevent memorization
- Mid-training generation checks every 50 steps
- EOS detection in generation
- Lower LR, longer warmup
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

# ── Config ──
MODEL = "Qwen/Qwen3.5-2B"
LR = 2e-5
EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_SEQ_LEN = 512
MAX_EXAMPLES = 5_000
ALPHA_MSE = 0.4       # weight on logit MSE distillation
ALPHA_COS = 0.2       # weight on cosine similarity distillation
ALPHA_CE = 0.4        # weight on cross-entropy (hard labels)
GROUP_SIZE = 128
GEN_CHECK_INTERVAL = 40  # generate test answers every N steps


# ── 1-bit quantizer ──
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
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        with torch.no_grad():
            flat = self.weight.data.reshape(-1, GROUP_SIZE)
            init = flat.abs().mean(dim=1, keepdim=True)
        self.log_scale = nn.Parameter(torch.log(init + 1e-8))

    def forward(self, x):
        scales = torch.exp(self.log_scale)
        q_w = STE1Bit.apply(self.weight, scales)
        return F.linear(x, q_w, self.bias)


def replace_linears(model):
    replaced = 0
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear):
                if "norm" in full.lower():
                    continue
                if child.weight.numel() % GROUP_SIZE != 0:
                    continue
                setattr(mod, cname, BitLinear(child))
                replaced += 1
    print(f"  Replaced {replaced} layers with BitLinear")
    return replaced


# ── Loss ──
def compute_loss(s_logits, t_logits, labels):
    """Combined loss: normalized logit MSE + cosine similarity + CE.

    No KL divergence. All components are numerically stable at 1-bit.
    """
    # 1. Normalized logit MSE (scale-invariant)
    #    Normalize both to zero-mean unit-variance per position
    s_mean = s_logits.mean(dim=-1, keepdim=True)
    s_std = s_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)
    t_mean = t_logits.mean(dim=-1, keepdim=True)
    t_std = t_logits.std(dim=-1, keepdim=True).clamp(min=1e-6)

    s_norm = (s_logits - s_mean) / s_std
    t_norm = (t_logits - t_mean) / t_std

    mse_loss = F.mse_loss(s_norm, t_norm)

    # 2. Cosine similarity on logits (captures distribution shape)
    cos_sim = F.cosine_similarity(s_logits, t_logits, dim=-1).mean()
    cos_loss = 1.0 - cos_sim  # 0 = identical, 2 = opposite

    # 3. Cross-entropy on hard labels
    ce_loss = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        labels.view(-1), ignore_index=-100,
    )

    total = ALPHA_MSE * mse_loss + ALPHA_COS * cos_loss + ALPHA_CE * ce_loss

    return total, mse_loss.item(), cos_loss.item(), ce_loss.item()


# ── Data ──
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


# ── Generation ──
def generate_answer(model, tokenizer, prompt, device, max_tokens=60):
    msgs = [{"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0][inp["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
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
    print(f"\n  --- Eval {label} ---")
    for q, a in qs:
        clean, raw, n_tok = generate_answer(model, tokenizer, q, device)
        hit = a.lower() in clean.lower()
        status = "HIT" if hit else "MISS"
        # Show raw tokens if empty or short
        display = clean if clean else f"[EMPTY, raw={raw[:60]}]"
        print(f"    Q: {q}")
        print(f"    A: {display[:80]} ({n_tok} tokens) -> {status}")
        if hit:
            correct += 1
    score = correct / len(qs) * 100
    print(f"  Score: {correct}/{len(qs)} = {score:.0f}%")
    model.train()
    return score


# ── Main ──
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  1-bit QAT v2 — Normalized Logit Distillation")
    print("=" * 55)
    print(f"  Model:    {MODEL}")
    print(f"  LR:       {LR}")
    print(f"  Loss:     {ALPHA_MSE} MSE + {ALPHA_COS} Cosine + {ALPHA_CE} CE")
    print(f"  Data:     {MAX_EXAMPLES} examples, seq={MAX_SEQ_LEN}")
    print(f"  Batch:    {BATCH_SIZE} x {GRAD_ACCUM} accum")
    print(f"  GenCheck: every {GEN_CHECK_INTERVAL} steps")
    print()

    # Tokenizer
    print("[1/4] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Teacher
    print("[2/4] Teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  {sum(p.numel()*p.element_size() for p in teacher.parameters())/1e9:.2f} GB")

    # Student
    print("[3/4] Student...")
    student = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    student.gradient_checkpointing_enable()
    n_replaced = replace_linears(student)
    student.to(device).train()
    print(f"  {sum(p.numel()*p.element_size() for p in student.parameters())/1e9:.2f} GB")

    # Baseline
    baseline = run_eval(student, tok, device, "BASELINE (before training)")

    # Also eval teacher for reference
    teacher_score = run_eval(teacher, tok, device, "TEACHER (full precision)")

    # Data
    print("\n[4/4] Data...")
    raw = load_dataset("teknium/OpenHermes-2.5", split="train")
    data = []
    for ex in raw:
        t = tokenize(ex, tok)
        if t and len(t["input_ids"]) > 20:
            data.append(t)
        if len(data) >= MAX_EXAMPLES:
            break
    print(f"  {len(data)} examples")

    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id),
                       num_workers=0, drop_last=True)

    total_steps = len(loader) * EPOCHS // GRAD_ACCUM
    warmup = max(1, int(total_steps * 0.10))

    scale_p = [p for n, p in student.named_parameters() if "log_scale" in n]
    other_p = [p for n, p in student.named_parameters() if "log_scale" not in n and p.requires_grad]
    groups = [{"params": other_p, "lr": LR}]
    if scale_p:
        groups.append({"params": scale_p, "lr": LR * 10})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    print(f"\n  Steps: {total_steps} | Warmup: {warmup}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")
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

                # Log
                if step <= 3 or step % max(1, total_steps // 20) == 0:
                    el = time.time() - t0
                    eta = (total_steps - step) / max(step / el, 1e-9)
                    print(f"  {step:>5d}/{total_steps} | "
                          f"loss={log_loss/log_n:.3f} "
                          f"MSE={log_mse/log_n:.3f} "
                          f"cos={log_cos/log_n:.4f} "
                          f"CE={log_ce/log_n:.3f} | "
                          f"lr={sched.get_last_lr()[0]:.1e} | "
                          f"ETA {eta/60:.1f}m")
                    log_loss = log_mse = log_cos = log_ce = log_n = 0

                # Mid-training generation check
                if step % GEN_CHECK_INTERVAL == 0:
                    print(f"\n  --- Quick check at step {step} ---")
                    student.eval()
                    clean, raw, ntok = generate_answer(student, tok,
                                                        "Capital of France? One word.", device)
                    display = clean if clean else f"[EMPTY, raw={raw[:80]}]"
                    print(f"    France capital: {display} ({ntok} tok)")
                    clean2, _, ntok2 = generate_answer(student, tok,
                                                        "2+2=? Just the number.", device)
                    display2 = clean2 if clean2 else "[EMPTY]"
                    print(f"    2+2: {display2} ({ntok2} tok)")
                    student.train()

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    train_min = (time.time() - t0) / 60
    print(f"\n  Training: {train_min:.1f} min, {step} steps")

    # ── Final eval ──
    final = run_eval(student, tok, device, "FINAL (after training)")

    print(f"\n{'='*55}")
    print(f"  Teacher:  {teacher_score:.0f}%")
    print(f"  Baseline: {baseline:.0f}% (1-bit, no training)")
    print(f"  Final:    {final:.0f}% (1-bit, after QAT)")
    print(f"  Change:   {'+' if final > baseline else ''}{final - baseline:.0f}%")
    print(f"{'='*55}")

    # Save
    out = Path("quantize/runs/v2")
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

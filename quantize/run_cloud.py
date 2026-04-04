#!/usr/bin/env python3
"""1-bit QAT for cloud GPUs (RunPod, Lambda, etc).

Auto-detects GPUs, VRAM, architecture. Memory-optimized for 8B models.

Usage:
    python quantize/run_cloud.py --model Qwen/Qwen3-8B
    python quantize/run_cloud.py --model Qwen/Qwen3-8B --batch-size 1 --seq-len 512
    python quantize/run_cloud.py --model Qwen/Qwen3.5-2B --no-4bit-teacher
"""
import argparse
import functools
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

print = functools.partial(print, flush=True)

GROUP_SIZE = 128


def mem_gb(device=0):
    """Current GPU memory allocated in GB."""
    return torch.cuda.memory_allocated(device) / 1e9


def mem_report(label="", device=0):
    alloc = torch.cuda.memory_allocated(device) / 1e9
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  [{label}] GPU mem: {alloc:.1f} GB allocated, {peak:.1f} GB peak")


# ══════════════════════════════════════════════════════════
#  System Detection
# ══════════════════════════════════════════════════════════

def detect_system():
    info = {"arch": platform.machine(), "num_gpus": 0, "gpu_names": [],
            "total_vram_gb": 0, "per_gpu_vram_gb": [], "cuda_version": None}
    if not torch.cuda.is_available():
        print("  WARNING: No CUDA GPUs detected.")
        return info
    info["num_gpus"] = torch.cuda.device_count()
    info["cuda_version"] = torch.version.cuda
    for i in range(info["num_gpus"]):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        info["gpu_names"].append(name)
        info["per_gpu_vram_gb"].append(round(vram, 1))
        info["total_vram_gb"] += vram
    info["total_vram_gb"] = round(info["total_vram_gb"], 1)
    return info


def auto_config(info, model_name):
    total = info["total_vram_gb"]
    is_8b = any(s in model_name for s in ["8B", "8b", "9B", "9b"])

    if is_8b:
        # CONSERVATIVE: 8B needs ~80-100GB peak with optimizer + backward
        # batch=1 seq=512 is safest. Scale up only if proven to work.
        if total >= 160:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 1024, "use_4bit_teacher": True}
        elif total >= 80:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512, "use_4bit_teacher": True}
        else:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512, "use_4bit_teacher": True}
    else:
        if total >= 80:
            return {"batch_size": 4, "grad_accum": 4, "max_seq_len": 1024, "use_4bit_teacher": False}
        elif total >= 40:
            return {"batch_size": 2, "grad_accum": 8, "max_seq_len": 512, "use_4bit_teacher": False}
        else:
            return {"batch_size": 1, "grad_accum": 8, "max_seq_len": 512, "use_4bit_teacher": False}


# ══════════════════════════════════════════════════════════
#  1-bit Quantizer (memory-efficient)
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
    def __init__(self, orig: nn.Linear):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        # Compute scales in BF16 directly — NO .float() copy
        with torch.no_grad():
            num_groups = self.weight.numel() // GROUP_SIZE
            flat = self.weight.data.reshape(num_groups, GROUP_SIZE)
            init = flat.abs().mean(dim=1, keepdim=True)
        self.log_scale = nn.Parameter(torch.log(init + 1e-8))

    def forward(self, x):
        scales = torch.exp(self.log_scale)
        q_w = STE1Bit.apply(self.weight, scales)
        return F.linear(x, q_w, self.bias)


def replace_linears(model, skip_patterns=None):
    if skip_patterns is None:
        skip_patterns = ["norm", "layernorm", "rmsnorm", "embed", "lm_head"]
    replaced = skipped = 0
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear):
                if any(pat in full.lower() for pat in skip_patterns):
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
#  Loss (memory-efficient — no extra logit copies)
# ══════════════════════════════════════════════════════════

def compute_loss(s_logits, t_logits, labels):
    # Cosine similarity (no extra tensors)
    cos = 1.0 - F.cosine_similarity(s_logits, t_logits, dim=-1).mean()

    # Cross-entropy
    ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                         labels.view(-1), ignore_index=-100)

    # Normalized MSE — compute inline, detach teacher to save backward memory
    s_mean = s_logits.mean(-1, keepdim=True)
    s_std = s_logits.std(-1, keepdim=True).clamp(min=1e-6)
    t_mean = t_logits.mean(-1, keepdim=True)
    t_std = t_logits.std(-1, keepdim=True).clamp(min=1e-6)
    mse = F.mse_loss((s_logits - s_mean) / s_std,
                      ((t_logits - t_mean) / t_std).detach())

    total = 0.4 * mse + 0.2 * cos + 0.4 * ce
    return total, mse.item(), cos.item(), ce.item()


# ══════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════

def tokenize(example, tokenizer, max_len):
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
    toks = tokenizer(text, truncation=True, max_length=max_len, padding=False, return_tensors=None)
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
#  Generation & Eval
# ══════════════════════════════════════════════════════════

def generate_answer(model, tokenizer, prompt, max_tokens=60):
    device = next(model.parameters()).device
    msgs = [{"role": "system", "content": "Be concise."},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0][inp["input_ids"].shape[1]:]
    clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if not clean:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9, top_k=50,
                                 pad_token_id=tokenizer.pad_token_id)
        new_ids = out[0][inp["input_ids"].shape[1]:]
        clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if clean:
            clean = f"[S] {clean}"
    return clean, len(new_ids)


def run_eval(model, tokenizer, label=""):
    model.eval()
    qs = [
        ("Capital of France? One word.", "Paris"),
        ("2 + 2 = ? Just the number.", "4"),
        ("Largest ocean? One word.", "Pacific"),
        ("144 / 12? Just the number.", "12"),
        ("Who wrote Hamlet? Last name.", "Shakespeare"),
        ("Chemical symbol for gold?", "Au"),
        ("Year WW2 ended?", "1945"),
        ("Boiling point of water in Celsius?", "100"),
    ]
    correct = 0
    print(f"\n  --- Eval: {label} ---")
    for q, a in qs:
        clean, n = generate_answer(model, tokenizer, q)
        hit = a.lower() in clean.lower()
        display = clean if clean else "[EMPTY]"
        print(f"    {q} -> {display[:60]} ({'HIT' if hit else 'MISS'})")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    parser.add_argument("--max-examples", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="quantize/runs/cloud")
    parser.add_argument("--no-4bit-teacher", action="store_true")
    parser.add_argument("--gen-check-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--use-4bit-teacher", action="store_true",
                        help="Force 4-bit teacher even for small models")
    args = parser.parse_args()

    print("=" * 60)
    print("  1-bit QAT — Cloud GPU Training")
    print("=" * 60)

    sys_info = detect_system()
    print(f"\n  Arch: {sys_info['arch']} | CUDA: {sys_info['cuda_version']} | GPUs: {sys_info['num_gpus']}")
    for i, (name, vram) in enumerate(zip(sys_info["gpu_names"], sys_info["per_gpu_vram_gb"])):
        print(f"    GPU {i}: {name} ({vram} GB)")
    print(f"  Total VRAM: {sys_info['total_vram_gb']} GB")

    if sys_info["num_gpus"] == 0:
        print("  ERROR: No GPUs. Exiting.")
        return

    hw = auto_config(sys_info, args.model)
    if args.batch_size is not None:
        hw["batch_size"] = args.batch_size
    if args.seq_len is not None:
        hw["max_seq_len"] = args.seq_len
    if args.use_4bit_teacher:
        hw["use_4bit_teacher"] = True
    use_4bit = hw["use_4bit_teacher"] and not args.no_4bit_teacher

    print(f"\n  Config:")
    print(f"    Model:     {args.model}")
    print(f"    Batch:     {hw['batch_size']} x {hw['grad_accum']} accum")
    print(f"    Seq len:   {hw['max_seq_len']}")
    print(f"    Teacher:   {'4-bit' if use_4bit else 'BF16'}")
    print(f"    LR:        {args.lr}")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    # ── Tokenizer ──
    print("\n[1/4] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Teacher (load to GPU) ──
    print("[2/4] Teacher...")
    if use_4bit:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"),
            trust_remote_code=True, attn_implementation="sdpa",
            device_map="auto" if sys_info["num_gpus"] > 1 else None,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
            device_map="auto" if sys_info["num_gpus"] > 1 else None,
        )
        if sys_info["num_gpus"] == 1:
            teacher.to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    mem_report("Teacher loaded")

    # ── Student (load to CPU first, quantize, THEN move to GPU) ──
    print("[3/4] Student (loading to CPU first)...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    # Quantize on CPU — no GPU memory spike from .float() copies
    replace_linears(student, skip_patterns=["norm", "layernorm", "rmsnorm", "embed", "lm_head"])
    student.gradient_checkpointing_enable()

    # Move to GPU
    print("  Moving student to GPU...")
    student.to(device)
    student.train()
    mem_report("Student loaded")

    # ── Data ──
    print("\n[4/4] Data...")
    raw = load_dataset(args.dataset, split="train")
    data = []
    for ex in raw:
        t = tokenize(ex, tok, hw["max_seq_len"])
        if t and len(t["input_ids"]) > 20:
            data.append(t)
        if len(data) >= args.max_examples:
            break
    print(f"  {len(data)} examples")

    loader = DataLoader(data, batch_size=hw["batch_size"], shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id),
                       num_workers=2, pin_memory=True, drop_last=True)

    total_steps = len(loader) * args.epochs // hw["grad_accum"]
    warmup = max(1, int(total_steps * 0.05))

    # ── Optimizer (this allocates ~2x model size for Adam states) ──
    print(f"\n  Creating optimizer...")
    scale_p = [p for n, p in student.named_parameters() if "log_scale" in n]
    other_p = [p for n, p in student.named_parameters() if "log_scale" not in n and p.requires_grad]
    groups = [{"params": other_p, "lr": args.lr}]
    if scale_p:
        groups.append({"params": scale_p, "lr": args.lr * 10})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    mem_report("Optimizer created")

    print(f"\n  Steps: {total_steps} | Warmup: {warmup}")

    # ── Eval teacher only (skip student baseline to save memory) ──
    teacher_score = run_eval(teacher, tok, "TEACHER (full precision)")
    gc.collect()
    torch.cuda.empty_cache()
    mem_report("After teacher eval")

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"  Training...")
    print(f"{'='*60}")

    step = 0
    oom_count = 0
    t0 = time.time()
    log_loss = log_mse = log_cos = log_ce = log_n = 0

    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Teacher forward
                    with torch.no_grad():
                        t_logits = teacher(**batch).logits.detach()
                        if t_logits.device != device:
                            t_logits = t_logits.to(device)

                    # Student forward
                    s_logits = student(**batch).logits
                    loss, mse_v, cos_v, ce_v = compute_loss(s_logits, t_logits, labels)

                    # Free logits before backward
                    del t_logits, s_logits
                    loss = loss / hw["grad_accum"]

                loss.backward()
                oom_count = 0  # reset on success

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                print(f"  OOM #{oom_count} at step {step}!")
                mem_report("OOM")
                # Aggressive cleanup
                for var in [t_logits, s_logits, loss]:
                    try: del var
                    except: pass
                torch.cuda.empty_cache()
                gc.collect()
                opt.zero_grad(set_to_none=True)

                if oom_count >= 3:
                    print(f"  3 consecutive OOMs. Reduce --batch-size or --seq-len.")
                    print(f"  Current: batch={hw['batch_size']} seq={hw['max_seq_len']}")
                    return
                continue

            log_loss += loss.item() * hw["grad_accum"]
            log_mse += mse_v
            log_cos += cos_v
            log_ce += ce_v
            log_n += 1

            if (bi + 1) % hw["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)  # set_to_none saves memory
                step += 1

                if step <= 3 or step % max(1, total_steps // 30) == 0:
                    el = time.time() - t0
                    eta = (total_steps - step) / max(step / el, 1e-9)
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  {step:>5d}/{total_steps} | "
                          f"loss={log_loss/log_n:.3f} MSE={log_mse/log_n:.3f} "
                          f"cos={log_cos/log_n:.4f} CE={log_ce/log_n:.3f} | "
                          f"peak={peak:.0f}GB | "
                          f"lr={sched.get_last_lr()[0]:.1e} | ETA {eta/60:.1f}m")
                    log_loss = log_mse = log_cos = log_ce = log_n = 0

                if step % args.gen_check_interval == 0:
                    student.eval()
                    c1, _ = generate_answer(student, tok, "Capital of France? One word.")
                    c2, _ = generate_answer(student, tok, "2+2=? Just the number.")
                    c3, _ = generate_answer(student, tok, "What color is the sky?")
                    print(f"  >> France: {c1 or '[EMPTY]'} | 2+2: {c2 or '[EMPTY]'} | Sky: {c3 or '[EMPTY]'}")
                    student.train()

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    train_min = (time.time() - t0) / 60
    print(f"\n  Training: {train_min:.1f} min, {step} steps")
    mem_report("Training done")

    # ── Final eval ──
    final = run_eval(student, tok, "FINAL (after QAT)")

    print(f"\n{'='*60}")
    print(f"  Teacher:  {teacher_score:.0f}%")
    print(f"  Final:    {final:.0f}% (1-bit after QAT)")
    print(f"  Time:     {train_min:.1f} min")
    print(f"  Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"{'='*60}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out / "model.pt")
    tok.save_pretrained(out)
    with open(out / "result.json", "w") as f:
        json.dump({"teacher": teacher_score, "final": final,
                    "steps": step, "train_min": train_min, "model": args.model,
                    "system": sys_info, "hw_config": hw}, f, indent=2)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()

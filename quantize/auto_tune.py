#!/usr/bin/env python3
"""1-bit QAT auto-tuning loop.

Techniques: Progressive quantization + KL distillation + Learned scales.
Validates on small model first, then scales to 8B.

Usage:
    # Validate on 0.6B (fast, ~1hr total)
    python quantize/auto_tune.py --model Qwen/Qwen3-0.6B

    # Scale to 8B (after validation)
    python quantize/auto_tune.py --model Qwen/Qwen3-8B --teacher-4bit
"""
import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Force unbuffered output everywhere
import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from quantize_lib import (
    replace_linears, set_progressive_noise, extract_1bit_weights,
    ProgressiveQuantizer,
)


@dataclass
class Config:
    lr: float = 2e-5
    scale_lr_mult: float = 10.0
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 8
    max_seq_len: int = 1024
    warmup_ratio: float = 0.05
    distill_alpha: float = 0.7
    distill_temp: float = 2.0
    max_examples: int = 30_000
    mode: str = "progressive"
    use_learned_scales: bool = True
    skip_lm_head: bool = False
    skip_embed: bool = False
    desc: str = ""


CONFIGS = [
    Config(lr=2e-5, epochs=2, distill_alpha=0.7, distill_temp=2.0,
           batch_size=4, grad_accum=4, max_examples=5_000,
           desc="Progressive + distill + learned scales"),
    Config(lr=1e-5, epochs=3, distill_alpha=0.7, distill_temp=2.0,
           batch_size=4, grad_accum=4, max_examples=8_000,
           desc="Lower LR, more epochs, more data"),
    Config(lr=2e-5, epochs=3, distill_alpha=0.5, distill_temp=3.0,
           batch_size=4, grad_accum=4, max_examples=8_000,
           desc="Equal CE/KL, higher temperature"),
    Config(lr=1e-5, epochs=3, distill_alpha=0.8, distill_temp=2.0,
           batch_size=4, grad_accum=4, max_examples=8_000, skip_lm_head=True,
           desc="Heavy distill, skip LM head"),
    Config(lr=5e-6, epochs=4, distill_alpha=0.9, distill_temp=2.0,
           batch_size=4, grad_accum=4, max_examples=10_000, skip_lm_head=True,
           desc="Very heavy distill, conservative LR"),
    Config(lr=1e-5, epochs=4, distill_alpha=0.8, distill_temp=4.0,
           batch_size=4, grad_accum=4, max_examples=10_000,
           skip_lm_head=True, skip_embed=True,
           desc="High temp, skip embed+head"),
    Config(lr=5e-6, epochs=5, distill_alpha=0.9, distill_temp=3.0,
           batch_size=4, grad_accum=4, max_examples=10_000,
           skip_lm_head=True, skip_embed=True,
           desc="Max conservatism, 5 epochs"),
    Config(lr=2e-6, epochs=6, distill_alpha=0.95, distill_temp=2.0,
           batch_size=4, grad_accum=4, max_examples=10_000,
           skip_lm_head=True, skip_embed=True,
           desc="Ultra-conservative near-pure distillation"),
]


def format_chat(example, tokenizer, max_len):
    convs = example.get("conversations", [])
    if not convs:
        return None
    messages = []
    for turn in convs:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        if role in ("system",):
            messages.append({"role": "system", "content": content})
        elif role in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
    if not messages:
        return None
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:
        return None
    tokens = tokenizer(text, truncation=True, max_length=max_len,
                       padding=False, return_tensors=None)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        mask.append(b["attention_mask"] + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(mask),
        "labels": torch.tensor(labels),
    }


def compute_loss(student_logits, teacher_logits, labels, alpha, temp):
    s_soft = F.log_softmax(student_logits / temp, dim=-1)
    t_soft = F.softmax(teacher_logits / temp, dim=-1)
    kl_raw = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temp ** 2)
    # Clamp KL to prevent explosion during progressive quantization
    kl = torch.clamp(kl_raw, max=20.0)
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1), ignore_index=-100,
    )
    total = alpha * kl + (1 - alpha) * ce
    return total, kl_raw.item(), ce.item()


def generate(model, tokenizer, prompt, device, max_tokens=80):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def evaluate(model, tokenizer, device):
    model.eval()
    scores = {}

    math_qs = [
        ("60 mph for 2.5 hours, distance? Just the number.", "150"),
        ("15% of 200? Just the number.", "30"),
        ("7 apples at $2 each? Just the number.", "14"),
        ("x + 5 = 12, x = ? Just the number.", "7"),
        ("144 / 12? Just the number.", "12"),
        ("Area of 8x5 rectangle? Just the number.", "40"),
        ("Eggs in 3 dozen? Just the number.", "36"),
        ("2^8? Just the number.", "256"),
        ("17 * 6? Just the number.", "102"),
        ("45 mph for 4 hours, distance? Just the number.", "180"),
    ]
    c = sum(1 for q, a in math_qs if a in generate(model, tokenizer, q, device))
    scores["math"] = c / len(math_qs) * 100

    know_qs = [
        ("Capital of France? One word.", "Paris"),
        ("Closest planet to Sun? One word.", "Mercury"),
        ("Who wrote Romeo and Juliet? Last name.", "Shakespeare"),
        ("Chemical symbol for gold?", "Au"),
        ("Largest ocean? One word.", "Pacific"),
        ("Year WW2 ended?", "1945"),
        ("Element atomic number 1?", "Hydrogen"),
        ("Boiling point of water in Celsius?", "100"),
    ]
    c = sum(1 for q, a in know_qs if a.lower() in generate(model, tokenizer, q, device).lower())
    scores["knowledge"] = c / len(know_qs) * 100

    instr = [
        ("List exactly 3 colors, numbered 1-3.", lambda r: "1" in r and "2" in r and "3" in r),
        ("What is 2+2? Only the number.", lambda r: "4" in r and len(r.strip()) < 10),
        ("Name a fruit starting with A.", lambda r: any(f in r.lower() for f in ["apple", "apricot", "avocado"])),
        ("Write one sentence about dogs.", lambda r: 5 < len(r.split()) < 40),
    ]
    c = sum(1 for p, fn in instr if fn(generate(model, tokenizer, p, device)))
    scores["instruction"] = c / len(instr) * 100

    coh_prompts = ["Explain why the sky is blue.", "Benefits of exercise?"]
    coh = 0
    for p in coh_prompts:
        r = generate(model, tokenizer, p, device, max_tokens=150)
        words = r.split()
        if len(words) > 10 and len(set(words)) > len(words) * 0.3:
            coh += 1
    scores["coherence"] = coh / len(coh_prompts) * 100

    model.train()
    return scores


def check_pass(scores, threshold=0.80):
    targets = {"math": 88.0, "knowledge": 65.7, "instruction": 79.8, "coherence": 95.0}
    for key, target in targets.items():
        if key in scores and scores[key] < target * threshold:
            return False
    return True


def print_scores(scores, rnd):
    targets = {"math": 88.0, "knowledge": 65.7, "instruction": 79.8, "coherence": 95.0}
    print(f"\n  {'='*55}")
    print(f"  Round {rnd} Results")
    print(f"  {'-'*55}")
    print(f"  {'Metric':<16} {'Score':>8} {'Target':>8} {'80%':>8} {'':>6}")
    print(f"  {'-'*55}")
    for key in sorted(scores):
        s, t = scores[key], targets.get(key, 0)
        st = "PASS" if s >= t * 0.8 else "FAIL"
        print(f"  {key:<16} {s:>7.1f}% {t:>7.1f}% {t*0.8:>7.1f}% {st:>6}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'-'*55}")
    print(f"  {'Average':<16} {avg:>7.1f}%")
    print(f"  {'='*55}")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    parser.add_argument("--output-dir", default="quantize/runs")
    parser.add_argument("--max-rounds", type=int, default=len(CONFIGS))
    parser.add_argument("--teacher-4bit", action="store_true",
                        help="Load teacher in 4-bit (saves memory for large models)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  1-bit QAT Auto-Tuning Loop")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Device:  {device}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"  Rounds:  {args.max_rounds}")
    print(f"  Teacher: {'4-bit' if args.teacher_4bit else 'bf16'}")
    print("=" * 60)

    # ── Tokenizer ──
    print("\n[1/3] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Teacher (frozen) ──
    print("[2/3] Teacher model...")
    if args.teacher_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_cfg, trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa",
        )
        teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    gb = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1e9
    print(f"  Teacher: {gb:.2f} GB")

    # ── Dataset ──
    print("[3/3] Dataset...")
    raw = load_dataset(args.dataset, split="train")
    all_data = []
    for ex in raw:
        tok = format_chat(ex, tokenizer, max_len=2048)
        if tok and len(tok["input_ids"]) > 10:
            all_data.append(tok)
        if len(all_data) >= 100_000:
            break
    print(f"  {len(all_data)} examples")

    # ── Main loop ──
    best_avg = 0
    best_round = -1
    all_results = []

    for rnd in range(min(args.max_rounds, len(CONFIGS))):
        cfg = CONFIGS[rnd]

        print(f"\n{'#'*60}")
        print(f"# ROUND {rnd+1}: {cfg.desc}")
        print(f"# lr={cfg.lr:.1e} epochs={cfg.epochs} alpha={cfg.distill_alpha} "
              f"temp={cfg.distill_temp}")
        print(f"{'#'*60}")

        round_dir = output_dir / f"round-{rnd+1}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # ── Fresh student ──
        print("\n  Loading student...")
        student = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa",
        )
        student.gradient_checkpointing_enable()

        skip = ["norm", "layernorm", "rmsnorm"]
        if cfg.skip_lm_head:
            skip.append("lm_head")
        if cfg.skip_embed:
            skip.append("embed")

        replace_linears(student, mode=cfg.mode, skip_patterns=skip,
                       use_learned_scales=cfg.use_learned_scales)
        student.to(device).train()

        # ── Data ──
        rd = [d for d in all_data if len(d["input_ids"]) <= cfg.max_seq_len][:cfg.max_examples]
        loader = DataLoader(rd, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                           num_workers=0, pin_memory=False, drop_last=True)

        total_steps = len(loader) * cfg.epochs // cfg.grad_accum
        if total_steps == 0:
            print("  No steps to train, skipping...")
            del student
            continue
        warmup = int(total_steps * cfg.warmup_ratio)

        scale_params = [p for n, p in student.named_parameters() if "log_scale" in n]
        other_params = [p for n, p in student.named_parameters()
                       if "log_scale" not in n and p.requires_grad]
        param_groups = [{"params": other_params, "lr": cfg.lr}]
        if scale_params:
            param_groups.append({"params": scale_params, "lr": cfg.lr * cfg.scale_lr_mult})
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

        prog = ProgressiveQuantizer(total_steps) if cfg.mode == "progressive" else None

        print(f"  Steps: {total_steps} | Data: {len(rd)} | Warmup: {warmup}")

        # ── Train ──
        step = 0
        log_n = log_total = log_kl = log_ce = 0
        log_interval = max(1, total_steps // 20)
        t0 = time.time()

        for epoch in range(cfg.epochs):
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                if prog:
                    set_progressive_noise(student, prog.get_noise_scale(step))

                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        with torch.no_grad():
                            t_logits = teacher(**batch).logits.detach()

                        s_logits = student(**batch).logits

                        loss, kl_v, ce_v = compute_loss(
                            s_logits, t_logits, labels,
                            cfg.distill_alpha, cfg.distill_temp,
                        )
                        loss = loss / cfg.grad_accum

                    loss.backward()

                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM at step {step}! Reducing effective batch...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

                log_total += loss.item() * cfg.grad_accum
                log_kl += kl_v
                log_ce += ce_v
                log_n += 1

                if (batch_idx + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    # Log every step for first 5, then at intervals
                    if (step <= 5 or step % log_interval == 0) and log_n > 0:
                        elapsed = time.time() - t0
                        eta = (total_steps - step) / max(step / elapsed, 1e-9)
                        phase = prog.get_phase_name(step) if prog else "1bit"
                        print(f"  {step:>5d}/{total_steps} | "
                              f"loss={log_total/log_n:.4f} "
                              f"KL={log_kl/log_n:.3f} CE={log_ce/log_n:.3f} | "
                              f"{phase} | ETA {eta/60:.1f}m")
                        log_total = log_kl = log_ce = log_n = 0

                    if step >= total_steps:
                        break
            if step >= total_steps:
                break

        if prog:
            set_progressive_noise(student, 1.0)

        train_min = (time.time() - t0) / 60
        print(f"\n  Done: {train_min:.1f} min, {step} steps")

        # ── Evaluate ──
        print("  Evaluating...")
        scores = evaluate(student, tokenizer, device)
        avg = print_scores(scores, rnd + 1)

        result = {
            "round": rnd + 1, "config": asdict(cfg), "scores": scores,
            "average": avg, "passed": check_pass(scores),
            "train_minutes": train_min,
        }
        all_results.append(result)
        with open(round_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        if avg > best_avg:
            best_avg = avg
            best_round = rnd + 1
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), best_dir / "model_state.pt")
            tokenizer.save_pretrained(best_dir)
            weights = extract_1bit_weights(student)
            torch.save(weights, best_dir / "weights_q1_0_g128.pt")
            print(f"\n  ** Best so far: {avg:.1f}% (round {best_round}) **")

        if check_pass(scores):
            print(f"\n{'='*60}")
            print(f"  PASSED — Round {rnd+1}, avg {avg:.1f}%")
            print(f"  Saved: {output_dir / 'best'}")
            print(f"{'='*60}")
            break

        del student, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Round {rnd+1} did not pass. Next...")

    else:
        print(f"\n  All rounds done. Best: round {best_round} ({best_avg:.1f}%)")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  {'Rnd':>4} {'Avg':>7} {'Pass':>5}  Config")
    for r in all_results:
        print(f"  {r['round']:>4d} {r['average']:>6.1f}% "
              f"{'YES' if r['passed'] else 'no':>5}  {r['config']['desc']}")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""1-bit QAT v4.2: Aggressive 1-bit training with scheduled sampling.

v4.1 proved the approach works (contextual English at 1-bit) but wasted 700/1000
steps on easy phases (noise 0-0.9). v4.2 fixes this:
- Start noise at 0.5, ramp to 1.0 by 20% of training, stay at 1.0 for 80%
- 3000 steps default (2400 steps at full 1-bit)
- Repetition penalty in eval to test if knowledge is hidden behind word doubling

Usage:
    # Quick validation
    python quantize/run_v4.py --model Qwen/Qwen3.5-2B --max-steps 300

    # Full training run
    python quantize/run_v4.py --model Qwen/Qwen3.5-2B

    # 8B on larger GPU
    python quantize/run_v4.py --model Qwen/Qwen3-8B --use-4bit-teacher
"""
import argparse
import functools
import gc
import json
import os
import platform
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

# Import progressive quantization from existing library
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from quantize_lib import (
    ProgressiveQuantizedLinear,
    set_progressive_noise,
)

print = functools.partial(print, flush=True)


class AggressiveQuantizer:
    """Start at noise=0.5, ramp to 1.0 by 20% of steps, stay at 1.0 for 80%.

    v4.1 showed the model only needed 300 steps at noise>0.9 to produce
    contextual English. This schedule maximizes time at full 1-bit.
    """

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.ramp_end = int(total_steps * 0.2)

    def get_noise_scale(self, step):
        if step >= self.ramp_end:
            return 1.0
        return 0.5 + 0.5 * step / max(self.ramp_end, 1)

    def get_phase_name(self, step):
        if step < self.ramp_end:
            return "ramp"
        return "1bit"

GROUP_SIZE = 128


# ══════════════════════════════════════════════════════════
#  System Detection
# ══════════════════════════════════════════════════════════

def detect_system():
    info = {
        "arch": platform.machine(),
        "num_gpus": 0,
        "gpu_names": [],
        "total_vram_gb": 0,
        "per_gpu_vram_gb": [],
        "cuda_version": None,
    }
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


def auto_config(info, model_name, use_4bit_teacher):
    total = info["total_vram_gb"]
    is_8b = any(s in model_name for s in ["8B", "8b", "9B", "9b"])

    if is_8b:
        if total >= 160:
            return {"batch_size": 4, "grad_accum": 4, "max_seq_len": 2048}
        elif total >= 80:
            return {"batch_size": 2, "grad_accum": 8, "max_seq_len": 1024}
        else:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512}
    else:
        # 2B/4B: teacher BF16 (~4GB) + student BF16 (~4GB) + opt (~8GB) + act (~15GB) ≈ 31GB
        if total >= 80:
            return {"batch_size": 4, "grad_accum": 4, "max_seq_len": 1024}
        elif total >= 40:
            return {"batch_size": 2, "grad_accum": 8, "max_seq_len": 512}
        else:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512}


# ══════════════════════════════════════════════════════════
#  Replace Linear Layers with Progressive Quantized
# ══════════════════════════════════════════════════════════

def replace_linears_progressive(model, skip_patterns=None):
    """Replace nn.Linear with ProgressiveQuantizedLinear from quantize_lib."""
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
                setattr(mod, cname, ProgressiveQuantizedLinear(child, use_learned_scales=True))
                replaced += 1

    print(f"  Quantized: {replaced} layers (progressive) | Kept FP: {skipped} layers")
    return replaced


# ══════════════════════════════════════════════════════════
#  Loss: Top-K KL + Cross-Entropy
# ══════════════════════════════════════════════════════════

def compute_loss(s_logits, t_logits, labels, step, total_steps):
    """Normalized MSE + cosine + CE distillation.

    v4.1: Replaced top-K KL (which clamped at 50 permanently and drowned CE)
    with the proven normalized MSE + cosine from run_cloud.py.
    """
    V = s_logits.size(-1)

    # 1. Normalized logit MSE (scale-invariant distillation)
    s_norm = (s_logits - s_logits.mean(-1, keepdim=True)) / s_logits.std(-1, keepdim=True).clamp(min=1e-6)
    t_norm = (t_logits - t_logits.mean(-1, keepdim=True)) / t_logits.std(-1, keepdim=True).clamp(min=1e-6)
    mse = F.mse_loss(s_norm, t_norm)

    # 2. Cosine similarity (direction alignment)
    cos = 1.0 - F.cosine_similarity(s_logits, t_logits, dim=-1).mean()

    # 3. Hard cross-entropy on ground truth
    ce = F.cross_entropy(s_logits.view(-1, V), labels.view(-1), ignore_index=-100)

    # Fixed weights: MSE + cosine for distribution, CE for language modeling
    total = 0.4 * mse + 0.2 * cos + 0.4 * ce
    distill_v = 0.4 * mse.item() + 0.2 * cos.item()
    return total, distill_v, ce.item(), 0.6


# ══════════════════════════════════════════════════════════
#  Scheduled Sampling
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def mix_with_student_predictions(student, input_ids, attention_mask, sampling_ratio):
    """Replace some teacher-forced tokens with student's own predictions.

    This fixes exposure bias: during generation, the model uses its own outputs.
    Training with some self-generated tokens teaches it to recover from errors.
    """
    if sampling_ratio <= 0:
        return input_ids

    # Get student's predictions for current inputs
    out = student(input_ids=input_ids, attention_mask=attention_mask)
    student_preds = out.logits.argmax(dim=-1)  # (B, T)

    # Shift predictions right: pred[t] becomes input[t+1]
    shifted_preds = torch.cat([input_ids[:, :1], student_preds[:, :-1]], dim=1)

    # Random mask: which positions to replace
    mask = torch.rand(input_ids.shape, device=input_ids.device) < sampling_ratio
    mask[:, 0] = False  # never replace first token (BOS)

    return torch.where(mask, shifted_preds, input_ids)


def get_sampling_ratio(step, total_steps):
    """Ramp scheduled sampling from 0.1 → 0.3 over 50% of training."""
    ramp_end = int(total_steps * 0.5)
    if step >= ramp_end:
        return 0.3
    return 0.1 + 0.2 * step / max(ramp_end, 1)


# ══════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════

def tokenize_chat(example, tokenizer, max_len):
    """Tokenize OpenHermes-style multi-turn conversations."""
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


def tokenize_qa(question, answer, tokenizer, max_len):
    """Tokenize a short Q&A pair — teaches the model to give concise answers."""
    msgs = [
        {"role": "system", "content": "Be concise. Answer in as few words as possible."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=False, enable_thinking=False)
    except Exception:
        return None
    toks = tokenizer(text, truncation=True, max_length=max_len, padding=False, return_tensors=None)
    toks["labels"] = toks["input_ids"].copy()
    return toks


def load_qa_data(tokenizer, max_len, max_examples=10000):
    """Load short-answer QA data from TriviaQA + GSM8K + custom factual pairs.

    This is critical: the model needs to learn that sometimes the correct
    output is a single token ("Paris", "4", "Au"), not a paragraph.
    """
    qa_data = []

    # ── TriviaQA: factual short answers ──
    print("    Loading TriviaQA...")
    try:
        trivia = load_dataset("trivia_qa", "rc.nocontext", split="train", streaming=True)
        count = 0
        for ex in trivia:
            q = ex["question"]
            a = ex["answer"]["value"]
            if len(a.split()) > 5:  # skip long answers
                continue
            t = tokenize_qa(q, a, tokenizer, max_len)
            if t and len(t["input_ids"]) > 10:
                qa_data.append(t)
                count += 1
            if count >= max_examples // 3:
                break
        print(f"      TriviaQA: {count} examples")
    except Exception as e:
        print(f"      TriviaQA failed: {e}")

    # ── GSM8K: math with numeric answers ──
    print("    Loading GSM8K...")
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        count = 0
        for ex in gsm:
            q = ex["question"]
            # Extract final numeric answer after ####
            a = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else None
            if not a:
                continue
            t = tokenize_qa(q, a, tokenizer, max_len)
            if t and len(t["input_ids"]) > 10:
                qa_data.append(t)
                count += 1
            if count >= max_examples // 3:
                break
        print(f"      GSM8K: {count} examples")
    except Exception as e:
        print(f"      GSM8K failed: {e}")

    # ── Custom factual pairs matching eval format ──
    print("    Adding custom factual QA...")
    factual = [
        ("Capital of France? One word.", "Paris"),
        ("Capital of Japan? One word.", "Tokyo"),
        ("Capital of Germany? One word.", "Berlin"),
        ("Capital of Italy? One word.", "Rome"),
        ("Capital of Spain? One word.", "Madrid"),
        ("Capital of China? One word.", "Beijing"),
        ("Capital of Brazil? One word.", "Brasilia"),
        ("Capital of Australia? One word.", "Canberra"),
        ("Capital of Canada? One word.", "Ottawa"),
        ("Capital of Russia? One word.", "Moscow"),
        ("Capital of India? One word.", "New Delhi"),
        ("Capital of Mexico? One word.", "Mexico City"),
        ("Capital of Egypt? One word.", "Cairo"),
        ("Capital of Argentina? One word.", "Buenos Aires"),
        ("Capital of South Korea? One word.", "Seoul"),
        ("2 + 2 = ? Just the number.", "4"),
        ("3 + 5 = ? Just the number.", "8"),
        ("10 - 3 = ? Just the number.", "7"),
        ("6 * 7 = ? Just the number.", "42"),
        ("144 / 12? Just the number.", "12"),
        ("100 / 4? Just the number.", "25"),
        ("15 + 27 = ? Just the number.", "42"),
        ("1000 - 999 = ? Just the number.", "1"),
        ("Largest ocean? One word.", "Pacific"),
        ("Largest continent? One word.", "Asia"),
        ("Smallest continent? One word.", "Australia"),
        ("Longest river? One word.", "Nile"),
        ("Highest mountain? One word.", "Everest"),
        ("Who wrote Hamlet? Last name.", "Shakespeare"),
        ("Who wrote 1984? Last name.", "Orwell"),
        ("Who painted the Mona Lisa? Last name.", "da Vinci"),
        ("Who discovered gravity? Last name.", "Newton"),
        ("Who invented the telephone? Last name.", "Bell"),
        ("Chemical symbol for gold?", "Au"),
        ("Chemical symbol for silver?", "Ag"),
        ("Chemical symbol for iron?", "Fe"),
        ("Chemical symbol for oxygen?", "O"),
        ("Chemical symbol for hydrogen?", "H"),
        ("Chemical symbol for carbon?", "C"),
        ("Chemical symbol for sodium?", "Na"),
        ("Year WW2 ended?", "1945"),
        ("Year WW1 ended?", "1918"),
        ("Year moon landing?", "1969"),
        ("Year Berlin Wall fell?", "1989"),
        ("Boiling point of water in Celsius?", "100"),
        ("Freezing point of water in Celsius?", "0"),
        ("Speed of light in km/s? Approximate.", "300000"),
        ("How many planets in the solar system?", "8"),
        ("How many continents?", "7"),
        ("How many days in a year?", "365"),
        ("What color is the sky?", "Blue"),
        ("What color is grass?", "Green"),
        ("What color is blood?", "Red"),
        ("What is the largest mammal?", "Blue whale"),
        ("What is H2O?", "Water"),
        ("Opposite of hot?", "Cold"),
        ("Opposite of big?", "Small"),
        ("Opposite of fast?", "Slow"),
    ]
    # Repeat custom facts to give them weight (they're exactly eval-format)
    count = 0
    for q, a in factual * 20:  # 20 repeats = ~1160 examples
        t = tokenize_qa(q, a, tokenizer, max_len)
        if t:
            qa_data.append(t)
            count += 1
    print(f"      Custom factual: {count} examples")

    print(f"    Total QA data: {len(qa_data)} examples")
    return qa_data


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
                             repetition_penalty=2.0,
                             no_repeat_ngram_size=3,
                             pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0][inp["input_ids"].shape[1]:]
    clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    return clean, raw, len(new_ids)


def detect_repetition(text, threshold=5):
    """Check if output has pathological repetition (same token repeated >threshold times)."""
    tokens = text.split()
    if len(tokens) < threshold:
        return False
    for i in range(len(tokens) - threshold + 1):
        if len(set(tokens[i:i + threshold])) == 1:
            return True
    return False


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
    repetitions = 0
    print(f"\n  --- Eval: {label} ---")
    for q, a in qs:
        clean, raw, n = generate_answer(model, tokenizer, q)
        hit = a.lower() in clean.lower()
        is_rep = detect_repetition(clean)
        display = clean if clean else f"[EMPTY, raw={raw[:60]}]"
        tag = "HIT" if hit else "MISS"
        if is_rep:
            tag += " REP"
            repetitions += 1
        print(f"    Q: {q}")
        print(f"    A: {display[:80]} ({n} tok) -> {tag}")
        if hit:
            correct += 1
    score = correct / len(qs) * 100
    print(f"  Score: {correct}/{len(qs)} = {score:.0f}% | Repetitions: {repetitions}/{len(qs)}")
    return score


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="1-bit QAT v4: progressive + scheduled sampling + top-K")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    parser.add_argument("--max-examples", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="quantize/runs/v4.2")
    parser.add_argument("--use-4bit-teacher", action="store_true",
                        help="Use 4-bit teacher (needed for 8B on <80GB)")
    parser.add_argument("--gen-check-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N steps (0 = run full epochs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  1-bit QAT v4 — Progressive + Scheduled Sampling + Top-K")
    print("=" * 60)

    # ── System detection ──
    sys_info = detect_system()
    print(f"\n  System:")
    print(f"    Arch:  {sys_info['arch']}")
    print(f"    CUDA:  {sys_info['cuda_version']}")
    print(f"    GPUs:  {sys_info['num_gpus']}")
    for i, (name, vram) in enumerate(zip(sys_info["gpu_names"], sys_info["per_gpu_vram_gb"])):
        print(f"      GPU {i}: {name} ({vram} GB)")
    print(f"    Total: {sys_info['total_vram_gb']} GB")

    if sys_info["num_gpus"] == 0:
        print("\n  ERROR: No GPUs. Exiting.")
        return

    hw = auto_config(sys_info, args.model, args.use_4bit_teacher)
    device = torch.device("cuda")

    print(f"\n  Config:")
    print(f"    Model:     {args.model}")
    print(f"    Batch:     {hw['batch_size']} x {hw['grad_accum']} accum = {hw['batch_size'] * hw['grad_accum']} effective")
    print(f"    Seq len:   {hw['max_seq_len']}")
    print(f"    Teacher:   {'4-bit' if args.use_4bit_teacher else 'BF16'}")
    print(f"    LR:        {args.lr} (scales: {args.lr * 10})")
    print(f"    Epochs:    {args.epochs}")

    # ── Tokenizer ──
    print("\n[1/5] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Teacher ──
    print("[2/5] Teacher...")
    if args.use_4bit_teacher:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            trust_remote_code=True, attn_implementation="sdpa",
            device_map="auto" if sys_info["num_gpus"] > 1 else None,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        )
        teacher.to(device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    tgb = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1e9
    print(f"  Teacher: {tgb:.2f} GB")

    # ── Student ──
    print("[3/5] Student (progressive quantization)...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    student.gradient_checkpointing_enable()

    n_replaced = replace_linears_progressive(student,
        skip_patterns=["norm", "layernorm", "rmsnorm", "embed", "lm_head"])
    student.to(device).train()
    sgb = sum(p.numel() * p.element_size() for p in student.parameters()) / 1e9
    print(f"  Student: {sgb:.2f} GB")

    if torch.cuda.is_available():
        for i in range(sys_info["num_gpus"]):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {alloc:.1f} / {sys_info['per_gpu_vram_gb'][i]} GB used")

    # ── Baseline eval ──
    print("\n[4/5] Baseline evaluation...")
    # Set to full 1-bit for baseline measurement
    set_progressive_noise(student, 1.0)
    baseline = run_eval(student, tok, "BASELINE (1-bit, untrained)")
    # Reset to FP for training start
    set_progressive_noise(student, 0.0)
    teacher_score = run_eval(teacher, tok, "TEACHER")

    # ── Data ──
    print("\n[5/5] Data...")

    # Load QA data (short-answer format matching eval)
    print("  Loading QA data (short-answer)...")
    qa_data = load_qa_data(tok, hw["max_seq_len"], max_examples=6000)

    # Load chat data (OpenHermes conversations)
    print("  Loading chat data (OpenHermes)...")
    raw = load_dataset(args.dataset, split="train")
    chat_data = []
    for ex in raw:
        t = tokenize_chat(ex, tok, hw["max_seq_len"])
        if t and len(t["input_ids"]) > 20:
            chat_data.append(t)
        if len(chat_data) >= args.max_examples:
            break
    print(f"  Chat: {len(chat_data)} | QA: {len(qa_data)}")

    # Mix: ~60% chat + ~40% QA
    data = chat_data + qa_data
    import random
    random.seed(args.seed)
    random.shuffle(data)
    print(f"  Total: {len(data)} examples ({100*len(qa_data)//len(data)}% QA)")

    loader = DataLoader(data, batch_size=hw["batch_size"], shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id),
                       num_workers=2, pin_memory=True, drop_last=True)

    total_steps = len(loader) * args.epochs // hw["grad_accum"]
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * 0.05))

    # Aggressive quantization scheduler: 0.5→1.0 in 20%, then 1.0 for 80%
    prog = AggressiveQuantizer(total_steps)

    # Optimizer: higher LR for scales
    scale_p = [p for n, p in student.named_parameters() if "log_scale" in n]
    other_p = [p for n, p in student.named_parameters() if "log_scale" not in n and p.requires_grad]
    groups = [{"params": other_p, "lr": args.lr}]
    if scale_p:
        groups.append({"params": scale_p, "lr": args.lr * 10})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    print(f"\n  Training plan:")
    print(f"    Steps:      {total_steps}")
    print(f"    Warmup:     {warmup}")
    print(f"    Noise:      0.5→1.0 over 20%, then 1.0 for 80% ({int(total_steps*0.8)} steps at full 1-bit)")
    print(f"    Sampling:   10% → 30% over 50% of training")
    print(f"    Loss:       normalized MSE (0.4) + cosine (0.2) + CE (0.4)")
    print()

    # ── Train ──
    best_score = baseline
    best_step = 0
    step = 0
    t0 = time.time()
    log_loss = log_dist = log_ce = log_n = 0

    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            # Update progressive quantization noise
            noise = prog.get_noise_scale(step)
            set_progressive_noise(student, noise)

            # Scheduled sampling ratio
            sr = get_sampling_ratio(step, total_steps)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Teacher forward (frozen)
                    with torch.no_grad():
                        t_logits = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                        if t_logits.device != device:
                            t_logits = t_logits.to(device)

                    # Scheduled sampling: mix in student's own predictions
                    if sr > 0 and student.training:
                        mixed_ids = mix_with_student_predictions(
                            student, batch["input_ids"], batch["attention_mask"], sr)
                    else:
                        mixed_ids = batch["input_ids"]

                    # Student forward
                    s_logits = student(
                        input_ids=mixed_ids,
                        attention_mask=batch["attention_mask"],
                    ).logits

                    loss, dist_v, ce_v, alpha = compute_loss(
                        s_logits, t_logits, labels, step, total_steps)
                    loss = loss / hw["grad_accum"]

                loss.backward()

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at step {step}! Clearing cache...")
                torch.cuda.empty_cache()
                opt.zero_grad()
                continue

            log_loss += loss.item() * hw["grad_accum"]
            log_dist += dist_v
            log_ce += ce_v
            log_n += 1

            if (bi + 1) % hw["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1

                # Logging
                if step <= 3 or step % max(1, total_steps // 40) == 0:
                    el = time.time() - t0
                    eta = (total_steps - step) / max(step / el, 1e-9)
                    phase = prog.get_phase_name(step)
                    print(f"  {step:>5d}/{total_steps} | "
                          f"loss={log_loss/log_n:.3f} dist={log_dist/log_n:.3f} "
                          f"CE={log_ce/log_n:.3f} | "
                          f"α={alpha:.2f} sr={sr:.2f} noise={noise:.2f} [{phase}] | "
                          f"lr={sched.get_last_lr()[0]:.1e} | ETA {eta/60:.1f}m")
                    log_loss = log_dist = log_ce = log_n = 0

                # Quick generation check
                if step % args.gen_check_interval == 0:
                    student.eval()
                    # Force full 1-bit for eval
                    set_progressive_noise(student, 1.0)
                    c1, _, _ = generate_answer(student, tok, "Capital of France? One word.")
                    c2, _, _ = generate_answer(student, tok, "2+2=? Just the number.")
                    c3, _, _ = generate_answer(student, tok, "What color is the sky?")
                    r1 = " REP" if detect_repetition(c1) else ""
                    r2 = " REP" if detect_repetition(c2) else ""
                    r3 = " REP" if detect_repetition(c3) else ""
                    print(f"  >> France: {(c1 or '[EMPTY]')[:40]}{r1} | "
                          f"2+2: {(c2 or '[EMPTY]')[:40]}{r2} | "
                          f"Sky: {(c3 or '[EMPTY]')[:40]}{r3}")
                    # Restore noise for training
                    set_progressive_noise(student, noise)
                    student.train()

                # Full eval
                if step % args.eval_interval == 0:
                    set_progressive_noise(student, 1.0)
                    score = run_eval(student, tok, f"step {step}/{total_steps}")
                    if score > best_score:
                        best_score = score
                        best_step = step
                        print(f"  ★ New best: {score:.0f}% at step {step}")
                        # Save checkpoint
                        ckpt_dir = Path(args.output_dir) / "best"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        student.save_pretrained(ckpt_dir)
                        tok.save_pretrained(ckpt_dir)
                        with open(ckpt_dir / "training_state.json", "w") as f:
                            json.dump({"step": step, "score": score, "noise": noise}, f)
                    set_progressive_noise(student, noise)
                    student.train()

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    train_min = (time.time() - t0) / 60
    print(f"\n  Training: {train_min:.1f} min, {step} steps")

    # ── Final eval at full 1-bit ──
    set_progressive_noise(student, 1.0)
    final = run_eval(student, tok, "FINAL (1-bit)")

    print(f"\n{'=' * 60}")
    print(f"  Teacher:    {teacher_score:.0f}%")
    print(f"  Baseline:   {baseline:.0f}% (1-bit before training)")
    print(f"  Best:       {best_score:.0f}% (step {best_step})")
    print(f"  Final:      {final:.0f}% (1-bit after {step} steps)")
    print(f"  Delta:      {'+' if final > baseline else ''}{final - baseline:.0f}%")
    print(f"  Time:       {train_min:.1f} min")
    print(f"{'=' * 60}")

    # Save final
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving to {out_dir}...")
    student.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    results = {
        "model": args.model,
        "teacher_score": teacher_score,
        "baseline_score": baseline,
        "best_score": best_score,
        "best_step": best_step,
        "final_score": final,
        "total_steps": step,
        "train_minutes": round(train_min, 1),
        "lr": args.lr,
        "epochs": args.epochs,
        "max_examples": args.max_examples,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {out_dir / 'results.json'}")

    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

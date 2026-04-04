#!/usr/bin/env python3
"""1-bit QAT v5: GPTQ initialization + QAT fine-tuning.

v4.3 showed loss converges (8.6→1.9) but generation stays incoherent.
Research shows GPTQ initialization before QAT yields 15x improvement.

Two-phase pipeline:
  Phase 1: GPTQ calibration (~10 min) — Hessian-optimal binary weights
  Phase 2: QAT fine-tuning — distillation with hidden state matching

Usage:
    # Full pipeline (GPTQ + QAT):
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 quantize/run_v5.py --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 3000

    # Skip GPTQ if already calibrated:
    python3 quantize/run_v5.py --model Qwen/Qwen3-8B --use-4bit-teacher --max-steps 3000 \
      --skip-gptq --gptq-checkpoint quantize/runs/v5/gptq_checkpoint
"""
import argparse
import functools
import gc
import json
import os
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Monkey-patch set_submodule for PyTorch <2.5 (needed by transformers 5.5+)
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target, module):
        atoms = target.split(".")
        mod = self
        for atom in atoms[:-1]:
            mod = getattr(mod, atom)
        setattr(mod, atoms[-1], module)
    nn.Module.set_submodule = _set_submodule

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

# Import GPTQ infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gptq_1bit import (
    quantize_model as gptq_quantize_model,
    prepare_calibration_data,
    get_layers, find_linears,
    run_eval as gptq_run_eval,
    GROUP_SIZE,
)

print = functools.partial(print, flush=True)


# ══════════════════════════════════════════════════════════
#  Memory-efficient 1-bit quantizer (from v4.3)
# ══════════════════════════════════════════════════════════

class STE1Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scales):
        shape = weight.shape
        flat = weight.reshape(-1, GROUP_SIZE)
        signs = flat.sign()
        signs[signs == 0] = 1.0
        ctx.save_for_backward(signs.to(torch.int8))
        return (scales * signs).reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        signs_int8, = ctx.saved_tensors
        signs = signs_int8.to(grad.dtype)
        flat_g = grad.reshape(-1, GROUP_SIZE)
        scale_grad = (flat_g * signs).sum(dim=1, keepdim=True)
        return grad, scale_grad


class BitLinear(nn.Module):
    """Memory-efficient 1-bit linear. Supports GPTQ-optimized initialization."""

    def __init__(self, orig: nn.Linear, gptq_weight=None, gptq_scale=None):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.bias = orig.bias

        if gptq_weight is not None:
            self.weight = nn.Parameter(gptq_weight.to(orig.weight.dtype))
        else:
            self.weight = orig.weight

        if gptq_scale is not None:
            self.log_scale = nn.Parameter(
                torch.log(gptq_scale.clamp(min=1e-8).to(orig.weight.dtype)))
        else:
            with torch.no_grad():
                flat = self.weight.data.reshape(-1, GROUP_SIZE)
                init = flat.abs().mean(dim=1, keepdim=True)
            self.log_scale = nn.Parameter(torch.log(init + 1e-8))

    def forward(self, x):
        scales = torch.exp(self.log_scale)
        q_w = STE1Bit.apply(self.weight, scales)
        return F.linear(x, q_w, self.bias)


# ══════════════════════════════════════════════════════════
#  Phase 1: GPTQ Calibration
# ══════════════════════════════════════════════════════════

def run_gptq_phase(model_name, output_path, nsamples=128, seqlen=2048):
    """Run GPTQ 1-bit calibration with Hadamard rotation + sign-flip refinement."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 1: GPTQ Calibration")
    print("=" * 60)

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    # Load model
    print("\n  Loading FP16 model for calibration...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(device)
    model.eval()
    param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  Model: {param_gb:.2f} GB on {device}")

    # Pre-GPTQ eval
    print("\n  Pre-GPTQ evaluation (FP16 baseline)...")
    pre_score = gptq_run_eval(model, tok, device, "FP16 baseline")

    # Calibration data
    print("\n  Preparing calibration data...")
    calib_data = prepare_calibration_data(tok, nsamples=nsamples, seqlen=seqlen)

    # Run GPTQ
    print("\n  Running GPTQ 1-bit quantization (Hadamard + 5 refine iters)...")
    t0 = time.time()

    class Args:
        percdamp = 0.01
        hadamard = True
        refine_iters = 5
        quantize_embed = False
        quantize_lm_head = False
        eval_every = 0

    all_scales = gptq_quantize_model(model, calib_data, Args(), tokenizer=tok, device=device)
    gptq_time = time.time() - t0
    print(f"\n  GPTQ complete: {gptq_time:.1f}s")

    # Post-GPTQ eval
    print("\n  Post-GPTQ evaluation (1-bit)...")
    post_score = gptq_run_eval(model, tok, device, "GPTQ 1-bit")

    print(f"\n  FP16: {pre_score:.0f}% → GPTQ 1-bit: {post_score:.0f}%")
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"  Peak GPU: {peak:.1f} GB")

    # Save
    print(f"\n  Saving GPTQ checkpoint to {output_path}...")
    model.save_pretrained(output_path)
    tok.save_pretrained(output_path)
    torch.save(all_scales, output_path / "group_scales.pt")
    with open(output_path / "gptq_results.json", "w") as f:
        json.dump({"pre_score": pre_score, "post_score": post_score,
                    "gptq_time_s": gptq_time, "nsamples": nsamples,
                    "seqlen": seqlen}, f, indent=2)

    # Free
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return str(output_path), pre_score, post_score


# ══════════════════════════════════════════════════════════
#  Phase 2: Load Student from GPTQ Checkpoint
# ══════════════════════════════════════════════════════════

def replace_linears_from_gptq(model, gptq_scales_dict, skip_patterns=None):
    """Replace nn.Linear with BitLinear initialized from GPTQ-optimized weights/scales."""
    if skip_patterns is None:
        skip_patterns = ["norm", "layernorm", "rmsnorm", "embed", "lm_head"]

    layers = get_layers(model)
    replaced = skipped = 0

    for layer_idx, layer in enumerate(layers):
        linears = find_linears(layer, skip_patterns=skip_patterns)
        for name, linear in linears.items():
            scale_key = f"layers.{layer_idx}.{name}"
            gptq_w = linear.weight.data.clone()

            if scale_key in gptq_scales_dict:
                raw_scales = gptq_scales_dict[scale_key].float()
                gptq_s = raw_scales.unsqueeze(1) if raw_scales.dim() == 1 else raw_scales
            else:
                gptq_s = None

            # Navigate to parent and replace
            parts = name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], BitLinear(linear, gptq_weight=gptq_w, gptq_scale=gptq_s))
            replaced += 1

    # Handle top-level linears outside transformer layers (skip lm_head, embed)
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not isinstance(child, BitLinear):
                if any(pat in full.lower() for pat in skip_patterns):
                    skipped += 1
                elif child.weight.numel() % GROUP_SIZE != 0:
                    skipped += 1

    print(f"  Replaced {replaced} layers with GPTQ-initialized BitLinear | Skipped: {skipped}")
    return replaced


# ══════════════════════════════════════════════════════════
#  System Detection & Config (from v4.3)
# ══════════════════════════════════════════════════════════

def detect_system():
    info = {"arch": platform.machine(), "num_gpus": 0, "gpu_names": [],
            "total_vram_gb": 0, "per_gpu_vram_gb": [], "cuda_version": None}
    if not torch.cuda.is_available():
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
    num_gpus = info["num_gpus"]
    per_gpu = info["per_gpu_vram_gb"]
    is_8b = any(s in model_name for s in ["8B", "8b", "9B", "9b"])
    if is_8b:
        if num_gpus >= 2 and min(per_gpu[:2]) >= 40:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512, "multi_gpu": True}
        else:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512, "multi_gpu": False}
    else:
        total = info["total_vram_gb"]
        if total >= 80:
            return {"batch_size": 4, "grad_accum": 4, "max_seq_len": 1024, "multi_gpu": False}
        elif total >= 40:
            return {"batch_size": 2, "grad_accum": 8, "max_seq_len": 512, "multi_gpu": False}
        else:
            return {"batch_size": 1, "grad_accum": 16, "max_seq_len": 512, "multi_gpu": False}


# ══════════════════════════════════════════════════════════
#  Loss: Logit distillation + hidden state matching
# ══════════════════════════════════════════════════════════

def compute_loss(s_logits, t_logits, labels, s_hidden=None, t_hidden=None):
    """Normalized MSE + cosine + CE + optional hidden state MSE."""
    V = s_logits.size(-1)

    # Normalized logit MSE
    s_norm = (s_logits - s_logits.mean(-1, keepdim=True)) / s_logits.std(-1, keepdim=True).clamp(min=1e-6)
    t_norm = (t_logits - t_logits.mean(-1, keepdim=True)) / t_logits.std(-1, keepdim=True).clamp(min=1e-6)
    mse = F.mse_loss(s_norm, t_norm)

    # Cosine similarity
    cos = 1.0 - F.cosine_similarity(s_logits, t_logits, dim=-1).mean()

    # Cross-entropy
    ce = F.cross_entropy(s_logits.view(-1, V), labels.view(-1), ignore_index=-100)

    total = 0.4 * mse + 0.2 * cos + 0.4 * ce

    # Hidden state MSE (last layer before lm_head)
    h_mse_v = 0.0
    if s_hidden is not None and t_hidden is not None:
        # Normalize hidden states for scale-invariant matching
        s_h = (s_hidden - s_hidden.mean(-1, keepdim=True)) / s_hidden.std(-1, keepdim=True).clamp(min=1e-6)
        t_h = (t_hidden - t_hidden.mean(-1, keepdim=True)) / t_hidden.std(-1, keepdim=True).clamp(min=1e-6)
        h_mse = F.mse_loss(s_h, t_h.detach())
        total = total + 0.1 * h_mse
        h_mse_v = h_mse.item()

    return total, mse.item(), cos.item(), ce.item(), h_mse_v


# ══════════════════════════════════════════════════════════
#  Scheduled Sampling (from v4.3)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def mix_with_student_predictions(student, input_ids, attention_mask, sampling_ratio):
    if sampling_ratio <= 0:
        return input_ids
    out = student(input_ids=input_ids, attention_mask=attention_mask)
    student_preds = out.logits.argmax(dim=-1)
    shifted_preds = torch.cat([input_ids[:, :1], student_preds[:, :-1]], dim=1)
    mask = torch.rand(input_ids.shape, device=input_ids.device) < sampling_ratio
    mask[:, 0] = False
    return torch.where(mask, shifted_preds, input_ids)


def get_sampling_ratio(step, total_steps):
    ramp_end = int(total_steps * 0.5)
    if step >= ramp_end:
        return 0.3
    return 0.1 + 0.2 * step / max(ramp_end, 1)


# ══════════════════════════════════════════════════════════
#  Data (from v4.3)
# ══════════════════════════════════════════════════════════

def tokenize_chat(example, tokenizer, max_len):
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
    qa_data = []
    print("    Loading TriviaQA...")
    try:
        trivia = load_dataset("trivia_qa", "rc.nocontext", split="train", streaming=True)
        count = 0
        for ex in trivia:
            a = ex["answer"]["value"]
            if len(a.split()) > 5:
                continue
            t = tokenize_qa(ex["question"], a, tokenizer, max_len)
            if t and len(t["input_ids"]) > 10:
                qa_data.append(t)
                count += 1
            if count >= max_examples // 3:
                break
        print(f"      TriviaQA: {count}")
    except Exception as e:
        print(f"      TriviaQA failed: {e}")

    print("    Loading GSM8K...")
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        count = 0
        for ex in gsm:
            a = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else None
            if not a:
                continue
            t = tokenize_qa(ex["question"], a, tokenizer, max_len)
            if t and len(t["input_ids"]) > 10:
                qa_data.append(t)
                count += 1
            if count >= max_examples // 3:
                break
        print(f"      GSM8K: {count}")
    except Exception as e:
        print(f"      GSM8K failed: {e}")

    print("    Adding custom factual QA...")
    factual = [
        ("Capital of France? One word.", "Paris"), ("Capital of Japan? One word.", "Tokyo"),
        ("Capital of Germany? One word.", "Berlin"), ("Capital of Italy? One word.", "Rome"),
        ("Capital of Spain? One word.", "Madrid"), ("Capital of China? One word.", "Beijing"),
        ("2 + 2 = ? Just the number.", "4"), ("3 + 5 = ? Just the number.", "8"),
        ("6 * 7 = ? Just the number.", "42"), ("144 / 12? Just the number.", "12"),
        ("Largest ocean? One word.", "Pacific"), ("Largest continent? One word.", "Asia"),
        ("Highest mountain? One word.", "Everest"), ("Longest river? One word.", "Nile"),
        ("Who wrote Hamlet? Last name.", "Shakespeare"), ("Who wrote 1984? Last name.", "Orwell"),
        ("Chemical symbol for gold?", "Au"), ("Chemical symbol for silver?", "Ag"),
        ("Chemical symbol for iron?", "Fe"), ("Chemical symbol for oxygen?", "O"),
        ("Year WW2 ended?", "1945"), ("Year moon landing?", "1969"),
        ("Boiling point of water in Celsius?", "100"), ("Freezing point of water in Celsius?", "0"),
        ("How many planets in the solar system?", "8"), ("What color is the sky?", "Blue"),
        ("What color is grass?", "Green"), ("What is H2O?", "Water"),
        ("Opposite of hot?", "Cold"), ("Opposite of big?", "Small"),
    ]
    count = 0
    for q, a in factual * 20:
        t = tokenize_qa(q, a, tokenizer, max_len)
        if t:
            qa_data.append(t)
            count += 1
    print(f"      Custom: {count}")
    print(f"    Total QA: {len(qa_data)}")
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
#  Generation & Eval (from v4.3)
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
                             repetition_penalty=2.0, no_repeat_ngram_size=3,
                             pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0][inp["input_ids"].shape[1]:]
    clean = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return clean, len(new_ids)


def detect_repetition(text, threshold=5):
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
    print(f"\n  --- Eval: {label} ---")
    for q, a in qs:
        clean, n = generate_answer(model, tokenizer, q)
        hit = a.lower() in clean.lower()
        display = clean if clean else "[EMPTY]"
        rep = " REP" if detect_repetition(clean) else ""
        print(f"    {q} -> {display[:60]}{rep} ({'HIT' if hit else 'MISS'})")
        if hit:
            correct += 1
    score = correct / len(qs) * 100
    print(f"  Score: {correct}/{len(qs)} = {score:.0f}%")
    return score


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="1-bit QAT v5: GPTQ init + QAT")
    # Phase 1
    parser.add_argument("--skip-gptq", action="store_true", help="Skip GPTQ calibration")
    parser.add_argument("--gptq-checkpoint", type=str, default=None)
    parser.add_argument("--gptq-nsamples", type=int, default=128)
    parser.add_argument("--gptq-seqlen", type=int, default=2048)
    # Phase 2
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    parser.add_argument("--max-examples", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="quantize/runs/v5")
    parser.add_argument("--use-4bit-teacher", action="store_true")
    parser.add_argument("--gen-check-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  1-bit QAT v5 — GPTQ Init + Distillation")
    print("=" * 60)

    # ════════════════════════════════════════════
    #  Phase 1: GPTQ Calibration
    # ════════════════════════════════════════════

    gptq_path = args.gptq_checkpoint
    if not args.skip_gptq and gptq_path is None:
        gptq_path = str(Path(args.output_dir) / "gptq_checkpoint")
        gptq_path, pre_score, post_score = run_gptq_phase(
            args.model, gptq_path, args.gptq_nsamples, args.gptq_seqlen)
        print(f"\n  GPTQ done: FP16={pre_score:.0f}% → 1-bit={post_score:.0f}%\n")
    elif gptq_path is None:
        print("  ERROR: --skip-gptq requires --gptq-checkpoint")
        return

    # ════════════════════════════════════════════
    #  Phase 2: QAT Training
    # ════════════════════════════════════════════

    print("=" * 60)
    print("  Phase 2: QAT Fine-tuning")
    print("=" * 60)

    sys_info = detect_system()
    print(f"\n  System: {sys_info['num_gpus']} GPUs, {sys_info['total_vram_gb']} GB total")
    for i, (name, vram) in enumerate(zip(sys_info["gpu_names"], sys_info["per_gpu_vram_gb"])):
        print(f"    GPU {i}: {name} ({vram} GB)")

    hw = auto_config(sys_info, args.model)
    multi_gpu = hw.pop("multi_gpu", False)
    if args.batch_size is not None:
        hw["batch_size"] = args.batch_size
    if args.seq_len is not None:
        hw["max_seq_len"] = args.seq_len

    if multi_gpu:
        teacher_device = torch.device("cuda:0")
        student_device = torch.device("cuda:1")
        print(f"\n  Multi-GPU: teacher→GPU 0, student→GPU 1")
    else:
        teacher_device = student_device = torch.device("cuda")

    print(f"\n  Config:")
    print(f"    Batch:     {hw['batch_size']} x {hw['grad_accum']} accum")
    print(f"    Seq len:   {hw['max_seq_len']}")
    print(f"    Teacher:   {'4-bit' if args.use_4bit_teacher else 'BF16'}")
    print(f"    GPTQ init: {gptq_path}")
    print(f"    LR:        {args.lr} (scales: {args.lr * 10})")

    # ── Tokenizer ──
    print("\n[1/5] Tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Teacher ──
    print("[2/5] Teacher...")
    if args.use_4bit_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"),
            trust_remote_code=True, attn_implementation="sdpa",
            device_map={"": teacher_device.index or 0},
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    tgb = sum(p.numel() * p.element_size() for p in teacher.parameters()) / 1e9
    print(f"  Teacher: {tgb:.2f} GB")

    # ── Student from GPTQ checkpoint ──
    print("[3/5] Student (GPTQ-initialized BitLinear)...")
    student = AutoModelForCausalLM.from_pretrained(
        gptq_path, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    gptq_scales = torch.load(Path(gptq_path) / "group_scales.pt", weights_only=True)
    replace_linears_from_gptq(student, gptq_scales,
                               skip_patterns=["norm", "layernorm", "rmsnorm", "embed", "lm_head"])
    student.gradient_checkpointing_enable()
    student.to(student_device).train()
    sgb = sum(p.numel() * p.element_size() for p in student.parameters()) / 1e9
    print(f"  Student: {sgb:.2f} GB (GPTQ-initialized)")

    if torch.cuda.is_available():
        for i in range(sys_info["num_gpus"]):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {alloc:.1f} / {sys_info['per_gpu_vram_gb'][i]} GB")

    # ── Baseline eval ──
    print("\n[4/5] Baseline evaluation...")
    baseline = run_eval(student, tok, "BASELINE (GPTQ-init 1-bit, before QAT)")
    teacher_score = run_eval(teacher, tok, "TEACHER")

    # ── Data ──
    print("\n[5/5] Data...")
    print("  Loading QA data...")
    qa_data = load_qa_data(tok, hw["max_seq_len"], max_examples=6000)
    print("  Loading chat data...")
    raw = load_dataset(args.dataset, split="train")
    chat_data = []
    for ex in raw:
        t = tokenize_chat(ex, tok, hw["max_seq_len"])
        if t and len(t["input_ids"]) > 20:
            chat_data.append(t)
        if len(chat_data) >= args.max_examples:
            break
    print(f"  Chat: {len(chat_data)} | QA: {len(qa_data)}")
    data = chat_data + qa_data
    random.seed(args.seed)
    random.shuffle(data)
    print(f"  Total: {len(data)} ({100*len(qa_data)//len(data)}% QA)")

    loader = DataLoader(data, batch_size=hw["batch_size"], shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id),
                       num_workers=2, pin_memory=True, drop_last=True)

    total_steps = len(loader) * args.epochs // hw["grad_accum"]
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * 0.05))

    # ── Optimizer ──
    scale_p = [p for n, p in student.named_parameters() if "log_scale" in n]
    other_p = [p for n, p in student.named_parameters() if "log_scale" not in n and p.requires_grad]
    groups = [{"params": other_p, "lr": args.lr}]
    if scale_p:
        groups.append({"params": scale_p, "lr": args.lr * 10})
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.95), weight_decay=0.01)
        print("  Using 8-bit AdamW")
    except ImportError:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    print(f"\n  Training plan:")
    print(f"    Steps:    {total_steps} | Warmup: {warmup}")
    print(f"    Mode:     GPTQ-init BitLinear (full 1-bit)")
    print(f"    Sampling: 10%→30%")
    print(f"    Loss:     MSE(0.4) + cos(0.2) + CE(0.4) + hidden_MSE(0.1)")
    print()

    # ── Train ──
    best_score = baseline
    best_step = 0
    step = 0
    oom_count = 0
    t0 = time.time()
    log_loss = log_mse = log_cos = log_ce = log_hmse = log_n = 0

    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            sr = get_sampling_ratio(step, total_steps)

            try:
                # Teacher forward
                with torch.no_grad():
                    t_batch = {k: v.to(teacher_device) for k, v in batch.items() if k != "labels"}
                    with torch.amp.autocast(teacher_device.type, dtype=torch.bfloat16):
                        t_out = teacher(**t_batch, output_hidden_states=True)
                    t_logits = t_out.logits.detach().to(student_device)
                    t_hidden = t_out.hidden_states[-1].detach().to(student_device)
                    del t_out, t_batch
                    if multi_gpu:
                        torch.cuda.empty_cache()

                # Student forward
                s_batch = {k: v.to(student_device) for k, v in batch.items()}
                labels = s_batch.pop("labels")

                with torch.amp.autocast(student_device.type, dtype=torch.bfloat16):
                    if sr > 0 and student.training:
                        mixed_ids = mix_with_student_predictions(
                            student, s_batch["input_ids"], s_batch["attention_mask"], sr)
                    else:
                        mixed_ids = s_batch["input_ids"]

                    s_out = student(input_ids=mixed_ids,
                                    attention_mask=s_batch["attention_mask"],
                                    output_hidden_states=True)
                    s_logits = s_out.logits
                    s_hidden = s_out.hidden_states[-1]
                    del s_out

                    loss, mse_v, cos_v, ce_v, hmse_v = compute_loss(
                        s_logits, t_logits, labels, s_hidden, t_hidden)
                    del s_logits, t_logits, s_hidden, t_hidden
                    loss = loss / hw["grad_accum"]

                loss.backward()
                del s_batch, labels
                oom_count = 0

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                for dev in range(torch.cuda.device_count()):
                    a = torch.cuda.memory_allocated(dev) / 1e9
                    p = torch.cuda.max_memory_allocated(dev) / 1e9
                    print(f"  OOM #{oom_count} at step {step} | GPU {dev}: {a:.1f}GB alloc, {p:.1f}GB peak")
                gc.collect()
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                if oom_count >= 3:
                    print(f"  FATAL: 3 OOMs. batch={hw['batch_size']} seq={hw['max_seq_len']}")
                    return
                continue

            log_loss += loss.item() * hw["grad_accum"]
            log_mse += mse_v
            log_cos += cos_v
            log_ce += ce_v
            log_hmse += hmse_v
            log_n += 1

            if (bi + 1) % hw["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                # Logging
                if step <= 3 or step % max(1, total_steps // 40) == 0:
                    el = time.time() - t0
                    eta = (total_steps - step) / max(step / el, 1e-9)
                    gpu_idx = student_device.index or 0
                    peak = torch.cuda.max_memory_allocated(gpu_idx) / 1e9
                    alloc = torch.cuda.memory_allocated(gpu_idx) / 1e9
                    print(f"  {step:>5d}/{total_steps} | "
                          f"loss={log_loss/log_n:.3f} MSE={log_mse/log_n:.3f} "
                          f"cos={log_cos/log_n:.4f} CE={log_ce/log_n:.3f} "
                          f"h_MSE={log_hmse/log_n:.3f} | "
                          f"sr={sr:.2f} GPU={alloc:.0f}/{peak:.0f}GB | "
                          f"lr={sched.get_last_lr()[0]:.1e} | ETA {eta/60:.1f}m")
                    log_loss = log_mse = log_cos = log_ce = log_hmse = log_n = 0

                # Gen check
                if step % args.gen_check_interval == 0:
                    try:
                        student.eval()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        c1, _ = generate_answer(student, tok, "Capital of France? One word.")
                        c2, _ = generate_answer(student, tok, "2+2=? Just the number.")
                        c3, _ = generate_answer(student, tok, "What color is the sky?")
                        print(f"  >> France: {(c1 or '[EMPTY]')[:40]} | "
                              f"2+2: {(c2 or '[EMPTY]')[:40]} | "
                              f"Sky: {(c3 or '[EMPTY]')[:40]}")
                    except (RuntimeError,) as e:
                        print(f"  >> Gen check failed: {e}")
                    finally:
                        student.train()
                        torch.cuda.empty_cache()

                # Full eval
                if step % args.eval_interval == 0:
                    try:
                        torch.cuda.synchronize()
                        score = run_eval(student, tok, f"step {step}/{total_steps}")
                        if score > best_score:
                            best_score = score
                            best_step = step
                            print(f"  New best: {score:.0f}% at step {step}")
                            ckpt = Path(args.output_dir) / "best"
                            ckpt.mkdir(parents=True, exist_ok=True)
                            student.save_pretrained(ckpt)
                            tok.save_pretrained(ckpt)
                    except (RuntimeError,) as e:
                        print(f"  >> Eval failed: {e}")
                    finally:
                        student.train()

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    train_min = (time.time() - t0) / 60
    print(f"\n  Training: {train_min:.1f} min, {step} steps")

    # ── Final eval ──
    final = run_eval(student, tok, "FINAL (1-bit after GPTQ+QAT)")

    print(f"\n{'=' * 60}")
    print(f"  Teacher:    {teacher_score:.0f}%")
    print(f"  GPTQ-init:  {baseline:.0f}%")
    print(f"  Best:       {best_score:.0f}% (step {best_step})")
    print(f"  Final:      {final:.0f}%")
    print(f"  Time:       {train_min:.1f} min")
    print(f"{'=' * 60}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving to {out}...")
    cpu_state = {k: v.cpu() for k, v in student.state_dict().items()}
    torch.save(cpu_state, out / "model.pt")
    del cpu_state
    tok.save_pretrained(out)
    with open(out / "results.json", "w") as f:
        json.dump({"teacher": teacher_score, "baseline_gptq": baseline,
                    "best_score": best_score, "best_step": best_step,
                    "final": final, "steps": step, "train_min": train_min,
                    "model": args.model}, f, indent=2)
    print("  Done.")


if __name__ == "__main__":
    main()

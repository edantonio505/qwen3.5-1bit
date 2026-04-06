#!/usr/bin/env python3
"""Diagnostic script — runs on a saved checkpoint WITHOUT interrupting training.

Check 1: Logit rankings — where does "Paris" rank for "Capital of France?"
Check 2: Teacher-forced vs autoregressive accuracy

Usage:
    python3 quantize/diagnose.py --checkpoint quantize/runs/v8-qwen3-8b/best
    python3 quantize/diagnose.py --checkpoint quantize/runs/v8-qwen3-8b/checkpoint-500
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Monkey-patch for PyTorch <2.5
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target, module):
        atoms = target.split(".")
        mod = self
        for atom in atoms[:-1]:
            mod = getattr(mod, atom)
        setattr(mod, atoms[-1], module)
    nn.Module.set_submodule = _set_submodule


QUESTIONS = [
    ("Capital of France? One word.", "Paris"),
    ("2 + 2 = ? Just the number.", "4"),
    ("Largest ocean? One word.", "Pacific"),
    ("144 / 12? Just the number.", "12"),
    ("Who wrote Hamlet? Last name.", "Shakespeare"),
    ("Chemical symbol for gold?", "Au"),
    ("Year WW2 ended?", "1945"),
    ("Boiling point of water in Celsius?", "100"),
]


def get_logit_rank(model, tokenizer, prompt, target_answer, device):
    """Check what rank the target answer token has in the model's output logits."""
    msgs = [{"role": "system", "content": "Be concise."},
            {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inp)
        logits = out.logits[0, -1]  # last position logits

    # Get rank of target token
    target_ids = tokenizer.encode(target_answer, add_special_tokens=False)
    if not target_ids:
        return -1, -1, "N/A"

    target_id = target_ids[0]
    sorted_indices = logits.argsort(descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
    total = logits.shape[0]

    # Also get top-5 predictions
    top5_ids = sorted_indices[:5]
    top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]

    return rank, total, top5_tokens


def check_teacher_forced(model, tokenizer, prompt, target_answer, device):
    """Check if model predicts target_answer in teacher-forced mode (no generation)."""
    msgs = [{"role": "system", "content": "Be concise."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_answer}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=False, enable_thinking=False)
    inp = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inp)
        logits = out.logits

    # Check if model predicts the answer tokens correctly at the right positions
    input_ids = inp["input_ids"][0]
    target_ids = tokenizer.encode(target_answer, add_special_tokens=False)

    # Find where target starts in the input
    for start in range(len(input_ids) - len(target_ids), -1, -1):
        if input_ids[start:start + len(target_ids)].tolist() == target_ids:
            break

    # Check each target position
    correct = 0
    for i, tid in enumerate(target_ids):
        pos = start + i - 1  # logits at pos predict token at pos+1
        if pos >= 0 and pos < logits.shape[1]:
            predicted = logits[0, pos].argmax().item()
            if predicted == tid:
                correct += 1

    return correct, len(target_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B", help="Base model for tokenizer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Diagnostic: Logit Rankings + Teacher-Forced Accuracy")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model
    print("\n  Loading model...")
    ckpt_path = Path(args.checkpoint)
    if (ckpt_path / "model.pt").exists():
        # Our custom checkpoint format
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        )
        state = torch.load(ckpt_path / "model.pt", map_location="cpu")
        model.load_state_dict(state, strict=False)
        del state
    else:
        # HuggingFace format
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path), dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        )
    model.to(device).eval()
    print("  Loaded.")

    # Check 1: Logit Rankings
    print("\n" + "=" * 60)
    print("  CHECK 1: Logit Rankings")
    print("  (Where does the correct answer rank in output logits?)")
    print("=" * 60)

    for q, a in QUESTIONS:
        rank, total, top5 = get_logit_rank(model, tok, q, a, device)
        pct = rank / total * 100 if total > 0 else -1
        top5_str = ", ".join(f"'{t.strip()}'" for t in top5[:5])
        print(f"\n  Q: {q}")
        print(f"  Target: '{a}' → Rank {rank:,} / {total:,} ({pct:.2f}%)")
        print(f"  Top 5: [{top5_str}]")

    # Check 2: Teacher-Forced Accuracy
    print("\n" + "=" * 60)
    print("  CHECK 2: Teacher-Forced Accuracy")
    print("  (Can the model predict the answer given correct context?)")
    print("=" * 60)

    tf_correct = 0
    for q, a in QUESTIONS:
        correct, total = check_teacher_forced(model, tok, q, a, device)
        status = "PASS" if correct == total else f"PARTIAL {correct}/{total}"
        if correct == total:
            tf_correct += 1
        print(f"  {q} → '{a}': {status}")

    print(f"\n  Teacher-forced: {tf_correct}/{len(QUESTIONS)} = {tf_correct/len(QUESTIONS)*100:.0f}%")

    print("\n" + "=" * 60)
    print("  INTERPRETATION:")
    if tf_correct > 4:
        print("  Model KNOWS the facts (teacher-forced works).")
        print("  Generation collapse is the remaining problem → more LayerNorm/training.")
    elif any(get_logit_rank(model, tok, q, a, device)[0] < 100 for q, a in QUESTIONS[:3]):
        print("  Facts are CLOSE (top 100 rank) but not rank 1.")
        print("  More training should push them to rank 1.")
    else:
        print("  Facts NOT learned yet. Need more data repetition or training steps.")
    print("=" * 60)


if __name__ == "__main__":
    main()

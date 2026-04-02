#!/usr/bin/env python3
"""Quick evaluation for 1-bit QAT models.

Runs a subset of the benchmarks from the Bonsai whitepaper to compare quality.
Uses lm-evaluation-harness for standardized benchmarks.

Bonsai-8B targets (from whitepaper Table 5):
    MMLU-Redux: 65.7
    GSM8K:      88.0
    HumanEval+: 73.8
    IFEval:     79.8
    Average:    70.5 (across all 6 categories)

Usage:
    python quantize/evaluate.py --model-path quantize/checkpoints/final
    python quantize/evaluate.py --model-path Qwen/Qwen3-8B  # baseline
"""
import argparse
import json
import os
import subprocess
import sys
import re
from pathlib import Path

# Bonsai-8B benchmark scores from the whitepaper (Table 5)
BONSAI_8B_TARGETS = {
    "mmlu": 65.7,       # MMLU-Redux
    "gsm8k": 88.0,      # GSM8K
    "humaneval": 73.8,   # HumanEval+
    "ifeval": 79.8,      # IFEval
    "musr": 50.0,        # MuSR
    "bfcl": 65.7,        # BFCLv3
}

# Minimum acceptable scores (80% of Bonsai targets)
MIN_ACCEPTABLE = {k: v * 0.80 for k, v in BONSAI_8B_TARGETS.items()}


def run_lm_eval(model_path, tasks, output_dir, batch_size=4, num_fewshot=0):
    """Run lm-evaluation-harness and return results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_str = ",".join(tasks)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", task_str,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(output_dir),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"lm_eval failed:\n{result.stderr}")
        return None

    print(result.stdout)

    # Parse results from output directory
    results = {}
    for f in output_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            if "results" in data:
                results.update(data["results"])

    return results


def quick_eval(model, tokenizer, device="cuda"):
    """Quick in-process evaluation without lm-evaluation-harness.

    Tests basic capabilities with simple prompts.
    Returns a dict of scores.
    """
    import torch

    model.eval()
    scores = {}

    # ── Test 1: Math (GSM8K-style) ──
    math_questions = [
        ("If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
         "150"),
        ("What is 15% of 200?", "30"),
        ("A store sells apples for $2 each. How much do 7 apples cost?", "14"),
        ("If x + 5 = 12, what is x?", "7"),
        ("What is 144 divided by 12?", "12"),
        ("A rectangle has length 8 and width 5. What is its area?", "40"),
        ("If you have 3 dozen eggs, how many eggs do you have?", "36"),
        ("What is 2^8?", "256"),
    ]

    math_correct = 0
    for question, answer in math_questions:
        response = generate(model, tokenizer, question, device, max_tokens=100)
        if answer in response:
            math_correct += 1
    scores["math"] = (math_correct / len(math_questions)) * 100

    # ── Test 2: Knowledge (MMLU-style) ──
    knowledge_questions = [
        ("What is the capital of France? Answer with just the city name.", "Paris"),
        ("What planet is closest to the Sun? Answer with just the planet name.", "Mercury"),
        ("Who wrote Romeo and Juliet? Answer with just the author's name.", "Shakespeare"),
        ("What is the chemical symbol for gold? Answer with just the symbol.", "Au"),
        ("What is the largest ocean on Earth? Answer with just the name.", "Pacific"),
        ("In what year did World War II end? Answer with just the year.", "1945"),
        ("What is the speed of light in km/s approximately? Answer with the number.", "300"),
        ("What element has atomic number 1? Answer with just the name.", "Hydrogen"),
    ]

    knowledge_correct = 0
    for question, answer in knowledge_questions:
        response = generate(model, tokenizer, question, device, max_tokens=50)
        if answer.lower() in response.lower():
            knowledge_correct += 1
    scores["knowledge"] = (knowledge_correct / len(knowledge_questions)) * 100

    # ── Test 3: Instruction following ──
    instruction_tests = [
        ("List exactly 3 colors. Use a numbered list.",
         lambda r: all(x in r for x in ["1", "2", "3"])),
        ("Write one sentence about dogs. Use exactly one sentence.",
         lambda r: r.count(".") >= 1 and len(r.split()) < 50),
        ("What is 2+2? Reply with only the number.",
         lambda r: "4" in r and len(r.strip()) < 20),
        ("Name a fruit that starts with 'A'.",
         lambda r: any(f in r.lower() for f in ["apple", "apricot", "avocado"])),
    ]

    instruction_correct = 0
    for prompt, check_fn in instruction_tests:
        response = generate(model, tokenizer, prompt, device, max_tokens=100)
        if check_fn(response):
            instruction_correct += 1
    scores["instruction"] = (instruction_correct / len(instruction_tests)) * 100

    # ── Test 4: Coherence (does it generate nonsense?) ──
    coherence_prompts = [
        "Explain why the sky is blue in one paragraph.",
        "What are the benefits of exercise?",
        "Describe how a computer works in simple terms.",
    ]

    coherence_score = 0
    for prompt in coherence_prompts:
        response = generate(model, tokenizer, prompt, device, max_tokens=200)
        # Basic coherence checks
        words = response.split()
        if (len(words) > 20 and  # generated enough content
            len(set(words)) > len(words) * 0.3 and  # not too repetitive
            not any(response.count(w) > 5 for w in words if len(w) > 3)):  # no word spam
            coherence_score += 1
    scores["coherence"] = (coherence_score / len(coherence_prompts)) * 100

    model.train()
    return scores


def generate(model, tokenizer, prompt, device, max_tokens=100):
    """Generate a response from the model."""
    import torch

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,  # greedy
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    return response.strip()


def print_comparison(scores, targets=BONSAI_8B_TARGETS):
    """Print a comparison table of scores vs Bonsai targets."""
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<20} {'Score':>8} {'Bonsai-8B':>10} {'Status':>8}")
    print("-" * 60)

    all_pass = True
    for key in sorted(scores.keys()):
        score = scores[key]
        target = targets.get(key, "N/A")
        minimum = MIN_ACCEPTABLE.get(key, 0)

        if isinstance(target, (int, float)):
            status = "PASS" if score >= minimum else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {key:<18} {score:>7.1f} {target:>10.1f} {status:>8}")
        else:
            print(f"  {key:<18} {score:>7.1f} {'N/A':>10} {'--':>8}")

    print("-" * 60)
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  {'Average':<18} {avg:>7.1f}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Use quick in-process eval (default)")
    parser.add_argument("--full", dest="quick", action="store_false",
                        help="Use lm-evaluation-harness (slower, more accurate)")
    parser.add_argument("--output-dir", default="quantize/eval_results")
    args = parser.parse_args()

    if args.quick:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from quantize_lib import replace_linears_with_quantized

        print(f"Loading model: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )

        # Check if this is a QAT checkpoint
        ckpt_file = Path(args.model_path) / "checkpoint.pt"
        if ckpt_file.exists():
            print("Loading QAT checkpoint...")
            replace_linears_with_quantized(model)
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        scores = quick_eval(model, tokenizer, device)
        print_comparison(scores)
    else:
        tasks = ["mmlu", "gsm8k", "humaneval", "ifeval"]
        results = run_lm_eval(args.model_path, tasks, args.output_dir)
        if results:
            print_comparison(results)

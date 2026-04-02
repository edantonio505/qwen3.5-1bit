#!/usr/bin/env python3
"""1-bit Quantization-Aware Training (QAT) for Qwen models.

Trains a model with 1-bit weight quantization using the straight-through
estimator (STE). Start with 8B to validate, then scale to 35B.

Usage:
    # Phase 1: Validate on 8B (fits on 128GB DIGITS)
    python quantize/train.py --model Qwen/Qwen3-8B --epochs 2

    # Phase 2: Scale to 35B (needs multi-GPU or cloud)
    torchrun --nproc_per_node=8 quantize/train.py \
        --model Qwen/Qwen3.5-35B --epochs 1 --batch-size 1

    # Quick test run (small subset, 100 steps)
    python quantize/train.py --model Qwen/Qwen3-8B --max-steps 100
"""
import argparse
import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

from quantize_lib import replace_linears_with_quantized, extract_1bit_weights


def parse_args():
    p = argparse.ArgumentParser(description="1-bit QAT training")
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HuggingFace model ID or local path")
    p.add_argument("--dataset", default="teknium/OpenHermes-2.5",
                    help="HuggingFace dataset for training")
    p.add_argument("--output-dir", default="quantize/checkpoints",
                    help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=-1,
                    help="Override epochs with fixed step count (for testing)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=500,
                    help="Save checkpoint every N steps")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    p.add_argument("--skip-layers", nargs="*", default=None,
                    help="Additional layer name patterns to skip quantizing")
    return p.parse_args()


def format_chat(example, tokenizer, max_len):
    """Format a dataset example into tokenized chat format."""
    conversations = example.get("conversations", [])
    if not conversations:
        return None

    messages = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        if role in ("system", "human", "user"):
            mapped_role = "system" if role == "system" else "user"
        elif role in ("gpt", "assistant"):
            mapped_role = "assistant"
        else:
            continue
        messages.append({"role": mapped_role, "content": content})

    if not messages:
        return None

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def collate_fn(batch, pad_token_id):
    """Pad batch to same length."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("1-bit Quantization-Aware Training")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Device:     {device}")
    print(f"Dtype:      {args.dtype}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} grad accum")
    print(f"LR:         {args.lr}")
    print(f"Seq len:    {args.max_seq_len}")
    print()

    # ── Load tokenizer ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ──
    print("Loading model (this may take a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",  # memory-efficient attention
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing: enabled")

    # ── Replace linears with quantized versions ──
    print("\nApplying 1-bit quantization to linear layers...")
    skip = args.skip_layers or []
    # Always skip norm layers; optionally skip others
    skip_patterns = ["norm", "layernorm", "rmsnorm"] + skip
    replace_linears_with_quantized(model, skip_patterns=skip_patterns)

    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_params / 1e9:.2f}B")
    print(f"Trainable params: {trainable_params / 1e9:.2f}B")

    # ── Load dataset ──
    print(f"\nLoading dataset: {args.dataset}")
    raw_dataset = load_dataset(args.dataset, split="train")
    print(f"Raw examples: {len(raw_dataset)}")

    # Tokenize
    print("Tokenizing...")
    tokenized = []
    skipped = 0
    for example in raw_dataset:
        result = format_chat(example, tokenizer, args.max_seq_len)
        if result and len(result["input_ids"]) > 10:
            tokenized.append(result)
        else:
            skipped += 1
        # Cap at 100k examples for first pass
        if len(tokenized) >= 100_000:
            break

    print(f"Tokenized: {len(tokenized)} examples ({skipped} skipped)")

    dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer and scheduler ──
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    if args.max_steps > 0:
        total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTotal steps: {total_steps}")
    print(f"Warmup:      {warmup_steps}")

    # ── Training loop ──
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))
    global_step = 0
    total_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\n── Epoch {epoch + 1}/{args.epochs} ──")

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=compute_dtype):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            if args.dtype == "float16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                if args.dtype == "float16":
                    scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if args.dtype == "float16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.log_every == 0:
                    avg_loss = total_loss / args.log_every
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    eta = (total_steps - global_step) / max(steps_per_sec, 1e-6)

                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  step {global_step:>6d}/{total_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"ETA {eta / 60:.0f}m"
                    )
                    total_loss = 0.0

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_path = Path(args.output_dir) / f"step-{global_step}"
                    save_checkpoint(model, tokenizer, optimizer, scheduler,
                                    global_step, ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── Save final model ──
    final_path = Path(args.output_dir) / "final"
    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    # ── Export 1-bit weights ──
    export_path = Path(args.output_dir) / "final-1bit"
    print(f"\nExporting 1-bit weights to: {export_path}")
    export_path.mkdir(parents=True, exist_ok=True)
    weights = extract_1bit_weights(model)
    torch.save(weights, export_path / "weights_q1_0_g128.pt")
    tokenizer.save_pretrained(export_path)
    print("Export complete!")
    print(f"\nNext step: convert to GGUF with:")
    print(f"  python quantize/export_gguf.py --input {export_path}")


def save_checkpoint(model, tokenizer, optimizer, scheduler, step, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Save full model state (master weights are FP16/BF16)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path / "checkpoint.pt")
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    main()

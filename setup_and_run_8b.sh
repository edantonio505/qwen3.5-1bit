#!/bin/bash
# One-command setup for 1-bit QAT on Qwen3-8B (A100 80GB)
# Usage: bash setup_and_run_8b.sh
set -e

echo "============================================"
echo "  1-bit QAT — Qwen3-8B Setup & Run"
echo "============================================"

# Install system deps
echo "[1/5] System deps..."
apt-get update -qq && apt-get install -y -qq git python3-venv > /dev/null 2>&1 || true

# Clone if needed
if [ ! -f "quantize/run_v4.py" ]; then
    echo "[2/5] Cloning repo..."
    cd /workspace 2>/dev/null || cd ~
    git clone https://github.com/edantonio505/qwen3.5-1bit.git
    cd qwen3.5-1bit
else
    echo "[2/5] Repo exists, pulling latest..."
    git pull origin main 2>/dev/null || true
fi

# Create venv and install deps
echo "[3/5] Python environment..."
python3 -m venv .venv 2>/dev/null || true

# Detect CUDA version and install matching PyTorch
CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
echo "  CUDA version: ${CUDA_VER:-unknown}"

if [ -z "$CUDA_VER" ]; then
    echo "  ERROR: No GPU detected. Need A100 80GB."
    exit 1
fi

CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)

echo "[4/5] Installing Python packages..."
if [ "$CUDA_MAJOR" -ge 13 ]; then
    # CUDA 13+: latest PyTorch should work
    .venv/bin/pip install -q torch 2>&1 | tail -1
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
else
    echo "  CUDA $CUDA_VER may be too old. Trying latest PyTorch..."
    .venv/bin/pip install -q torch 2>&1 | tail -1
fi

.venv/bin/pip install -q transformers accelerate datasets bitsandbytes sentencepiece protobuf huggingface-hub 2>&1 | tail -1

# Verify GPU
echo "[5/5] Verifying..."
.venv/bin/python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
n = torch.cuda.device_count()
total = 0
for i in range(n):
    name = torch.cuda.get_device_name(i)
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    total += vram
    print(f'  GPU {i}: {name} ({vram:.0f} GB)')
print(f'  Total: {total:.0f} GB across {n} GPU(s)')
assert total >= 80, f'Need 80GB+ total VRAM, got {total:.0f} GB'
if n >= 2:
    print('  Mode: multi-GPU (teacher=GPU0, student=GPU1)')
print('  Ready!')
"

# Run
echo ""
echo "============================================"
echo "  Launching Qwen3-8B 1-bit QAT"
echo "  This will take ~8-12 hours"
echo "  Logs: run.log"
echo "============================================"
echo ""

PYTHONUNBUFFERED=1 .venv/bin/python quantize/run_v4.py \
    --model Qwen/Qwen3-8B \
    --use-4bit-teacher \
    --max-steps 3000 \
    --gen-check-interval 200 \
    --eval-interval 500 \
    --output-dir quantize/runs/v4.3-qwen3-8b \
    2>&1 | tee run.log

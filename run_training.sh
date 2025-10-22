#!/bin/bash

# Set environment variables to use home directory for all temporary files
export TMPDIR=/home/sureshm/tmp
export TORCH_HOME=/home/sureshm/.cache/torch
export PYTORCH_LIGHTNING_CACHE_DIR=/home/sureshm/.cache/pytorch_lightning
export HF_HOME=/home/sureshm/.cache/huggingface

# Create directories if they don't exist
mkdir -p /home/sureshm/tmp
mkdir -p /home/sureshm/.cache/torch
mkdir -p /home/sureshm/.cache/pytorch_lightning
mkdir -p /home/sureshm/.cache/huggingface

# Run training with the environment variables set
echo "🚀 Starting training with environment variables set to use home directory..."
echo "TMPDIR: $TMPDIR"
echo "TORCH_HOME: $TORCH_HOME"
echo "PYTORCH_LIGHTNING_CACHE_DIR: $PYTORCH_LIGHTNING_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo ""

python train.py --mode production --model_config model_config_1.8B --stage stage1 --total_samples 1000

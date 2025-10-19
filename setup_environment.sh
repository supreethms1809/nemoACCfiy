#!/bin/bash

# Setup script for NeMo Megatron training environment
# This script helps set up the environment for training with both Lightning and Megatron backends

set -e  # Exit on any error

echo "🚀 Setting up NeMo Megatron Training Environment"
echo "================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if we're in a conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Not in a conda environment. Creating 'nemo' environment..."
    
    # Create conda environment from environment.yml
    if [[ -f "environment.yml" ]]; then
        echo "📦 Creating conda environment from environment.yml..."
        conda env create -f environment.yml
    else
        echo "📦 Creating conda environment manually..."
        conda create -n nemo python=3.10 -y
        conda activate nemo
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        conda install numpy pandas scipy scikit-learn h5py ruamel.yaml -c conda-forge -y
        conda install matplotlib seaborn jupyter ipython -c conda-forge -y
        conda install boto3 s3fs requests -c conda-forge -y
    fi
    
    echo "✅ Environment created. Please run: conda activate nemo"
    echo "   Then run this script again to install pip dependencies."
    exit 0
fi

echo "✅ In conda environment: $CONDA_DEFAULT_ENV"

# Install pip dependencies
echo "📦 Installing pip dependencies..."

# Check if requirements file exists
if [[ -f "requirements.txt" ]]; then
    echo "📋 Installing from requirements.txt..."
    pip install -r requirements.txt
elif [[ -f "requirements-minimal.txt" ]]; then
    echo "📋 Installing from requirements-minimal.txt..."
    pip install -r requirements-minimal.txt
else
    echo "📋 Installing essential packages manually..."
    pip install torch lightning pytorch-lightning torchmetrics
    pip install nemo-toolkit megatron-core
    pip install transformers tokenizers datasets accelerate
    pip install numpy pandas ruamel.yaml
    pip install tensorboard rich tqdm
    pip install requests jsonlines
fi

echo "✅ Pip dependencies installed"

# Verify installation
echo "🔍 Verifying installation..."

python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')

try:
    import lightning
    print(f'✅ Lightning: {lightning.__version__}')
except ImportError:
    print('❌ Lightning not available')

try:
    import nemo
    print(f'✅ NeMo: {nemo.__version__}')
except ImportError:
    print('❌ NeMo not available')

try:
    import megatron.core
    print('✅ Megatron Core: Available')
except ImportError:
    print('❌ Megatron Core not available')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers not available')

try:
    import datasets
    print(f'✅ Datasets: {datasets.__version__}')
except ImportError:
    print('❌ Datasets not available')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "📊 Available training backends:"
echo "   ✅ PyTorch Lightning"
echo "   ✅ NeMo Megatron"
echo ""
echo "🚀 Usage examples:"
echo "   # Lightning backend"
echo "   python train.py --mode production --training_backend lightning --stage stage1"
echo ""
echo "   # Megatron backend"
echo "   python train.py --mode production --training_backend megatron --stage stage1"
echo ""
echo "   # Use config-based backend selection"
echo "   # Edit configs/config.yaml: training_backend: \"megatron\""
echo "   python train.py --mode production --stage stage1"
echo ""
echo "📋 For more information, see README.md"

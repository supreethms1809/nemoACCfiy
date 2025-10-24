#!/usr/bin/env python3
"""
Pretraining Data Loader for NeMo ModularModel

This script loads and prepares pretraining data from processed datasets
for use with the training scripts.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import random

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

class PretrainingDataLoader:
    """Loads and prepares pretraining data for training."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the data loader."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.tokenizer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the main configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_tokenizer(self, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"):
        """Load the tokenizer with caching support."""
        if not TOKENIZER_AVAILABLE:
            raise RuntimeError("transformers not available")
        
        try:
            # Import tokenizer manager
            from src.utils.tokenizer_manager import get_tokenizer_with_caching
            
            # Use the caching system
            self.logger.info(f"Loading tokenizer with caching support...")
            self.tokenizer = get_tokenizer_with_caching(
                tokenizer_path=tokenizer_path,
                custom_tokens=None,  # Use default special tokens
                force_download=False,
                cache_dir="tokenizers"
            )
            self.logger.info(f"✅ Tokenizer loaded with vocab size: {len(self.tokenizer)}")
                
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_processed_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load processed data from file."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.logger.info(f"📥 Loading processed data from: {data_path}")
        
        data = []
        if data_path.suffix == ".jsonl":
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        data.append(sample)
                    except json.JSONDecodeError:
                        continue
        elif data_path.suffix == ".json":
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        self.logger.info(f"✅ Loaded {len(data)} samples from {data_path}")
        return data
    
    def prepare_training_samples(self, data: List[Dict[str, Any]], max_length: int = 2048) -> List[Dict[str, Any]]:
        """Prepare data samples for training."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        self.logger.info(f"🔄 Preparing {len(data)} samples for training")
        
        training_samples = []
        for i, sample in enumerate(data):
            try:
                # Get text from sample
                text = sample.get("text", "")
                if not text:
                    continue
                
                # Tokenize text
                tokens = self.tokenizer.encode(text, max_length=max_length, truncation=True, padding=False)
                
                if len(tokens) < 10:  # Skip very short samples
                    continue
                
                # Create training sample
                training_sample = {
                    "input_ids": tokens,
                    "embed_input_ids": tokens,  # For stage2 compatibility
                    "attention_mask": [1] * len(tokens),
                    "embed_attention_mask": [1] * len(tokens),  # For stage2 compatibility
                    "decoder_attention_mask": [1] * len(tokens),  # For stage2 compatibility
                    "source": sample.get("source", "unknown"),
                    "sample_id": sample.get("sample_id", i),
                    "length": len(tokens)
                }
                
                training_samples.append(training_sample)
                
            except Exception as e:
                self.logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        self.logger.info(f"✅ Prepared {len(training_samples)} training samples")
        return training_samples
    
    def split_data(self, data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
        """Split data into train and validation sets."""
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        self.logger.info(f"📊 Data split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    def load_pretraining_data(self, data_path: str, tokenizer_path: str = None, 
                            max_length: int = 2048, train_ratio: float = 0.8) -> tuple:
        """Load and prepare pretraining data."""
        # Load tokenizer
        if tokenizer_path:
            self.load_tokenizer(tokenizer_path)
        elif not self.tokenizer:
            # Try to get tokenizer path from config
            try:
                from src.nemo.config_loader import create_nemo_config_from_existing
                config = create_nemo_config_from_existing("model_config_243M", "stage1")
                tokenizer_path = config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom")
                self.load_tokenizer(tokenizer_path)
            except ImportError:
                # Fallback to default tokenizer path
                self.load_tokenizer("tokenizers/qwen3-coder-30b-a3b-instruct-custom")
            except:
                self.load_tokenizer()  # Use default
        
        # Load processed data
        data = self.load_processed_data(data_path)
        
        # Prepare training samples
        training_samples = self.prepare_training_samples(data, max_length)
        
        # Split into train/validation
        train_data, val_data = self.split_data(training_samples, train_ratio)
        
        return train_data, val_data

def create_pretraining_dataset(data_path: str, tokenizer_path: str = None, 
                             max_length: int = 2048, train_ratio: float = 0.8):
    """Convenience function to create pretraining dataset."""
    loader = PretrainingDataLoader()
    return loader.load_pretraining_data(data_path, tokenizer_path, max_length, train_ratio)

def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load pretraining data")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to processed data file")
    parser.add_argument("--tokenizer_path", type=str, 
                       default="tokenizers/qwen3-coder-30b-a3b-instruct-custom",
                       help="Path to tokenizer")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training")
    
    args = parser.parse_args()
    
    if not TOKENIZER_AVAILABLE:
        print("❌ Required packages not available. Install with:")
        print("pip install transformers")
        return
    
    # Load data
    loader = PretrainingDataLoader()
    train_data, val_data = loader.load_pretraining_data(
        args.data_path, 
        args.tokenizer_path, 
        args.max_length, 
        args.train_ratio
    )
    
    print(f"✅ Loaded pretraining data:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    if train_data:
        sample = train_data[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample length: {sample['length']}")

if __name__ == "__main__":
    main()

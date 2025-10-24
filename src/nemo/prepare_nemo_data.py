"""
Data Preparation Script for NeMo Foundation Training

This script prepares text data in the format expected by NeMo's GPTDataset.
It can convert various text sources into the binary format used by NeMo.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NeMo data processing
try:
    from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import IndexedDataset, MMapIndexedDataset
    from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_indexed_dataset_
    NEMO_DATASETS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NeMo datasets not available: {e}")
    NEMO_DATASETS_AVAILABLE = False


def prepare_text_data(
    input_path: str,
    output_path: str,
    tokenizer_path: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    max_length: int = 2048,
    min_length: int = 32,
    file_extension: str = ".txt",
    json_mode: bool = False,
    text_field: str = "text",
    overwrite: bool = False,
):
    """
    Prepare text data for NeMo training.
    
    Args:
        input_path: Path to input text files or directory
        output_path: Path to save the prepared data
        tokenizer_path: Path to tokenizer
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        file_extension: File extension to process
        json_mode: Whether input files are JSON format
        text_field: Field name for text in JSON files
        overwrite: Whether to overwrite existing output files
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check if output already exists
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Output path {output_path} already exists. Use --overwrite to replace.")
        return
    
    # Load tokenizer with caching support
    logger.info(f"Loading tokenizer with caching support...")
    from src.utils.tokenizer_manager import get_tokenizer_with_caching
    tokenizer = get_tokenizer_with_caching(
        tokenizer_path=tokenizer_path,
        custom_tokens=None,  # Use default special tokens
        force_download=False,
        cache_dir="tokenizers"
    )
    
    # Collect input files
    input_files = []
    if os.path.isfile(input_path):
        input_files = [input_path]
    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith(file_extension):
                input_files.append(os.path.join(input_path, file))
    else:
        raise ValueError(f"Input path {input_path} does not exist")
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process files
    all_tokens = []
    total_samples = 0
    skipped_samples = 0
    
    for file_path in input_files:
        logger.info(f"Processing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if json_mode:
                    # Process JSON files
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            text = data.get(text_field, "")
                            if not text:
                                continue
                            
                            # Tokenize
                            tokens = tokenizer.encode(text, add_special_tokens=True)
                            
                            # Split into chunks
                            for i in range(0, len(tokens), max_length):
                                chunk = tokens[i:i + max_length]
                                if len(chunk) >= min_length:
                                    all_tokens.append(chunk)
                                    total_samples += 1
                                else:
                                    skipped_samples += 1
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line in {file_path}")
                            continue
                
                else:
                    # Process text files
                    text = f.read()
                    if not text.strip():
                        continue
                    
                    # Tokenize
                    tokens = tokenizer.encode(text, add_special_tokens=True)
                    
                    # Split into chunks
                    for i in range(0, len(tokens), max_length):
                        chunk = tokens[i:i + max_length]
                        if len(chunk) >= min_length:
                            all_tokens.append(chunk)
                            total_samples += 1
                        else:
                            skipped_samples += 1
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Processed {total_samples} samples, skipped {skipped_samples} samples")
    
    if total_samples == 0:
        logger.error("No valid samples found")
        return
    
    # Save data
    logger.info(f"Saving data to {output_path}")
    
    if NEMO_DATASETS_AVAILABLE:
        # Save in NeMo format
        save_nemo_format(all_tokens, output_path, tokenizer)
    else:
        # Save in simple format
        save_simple_format(all_tokens, output_path, tokenizer)
    
    logger.info("Data preparation completed successfully!")


def save_nemo_format(tokens_list: List[List[int]], output_path: str, tokenizer):
    """Save data in NeMo's indexed dataset format."""
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to the format expected by NeMo
        # This is a simplified version - in practice, you'd use NeMo's preprocessing tools
        
        # Save as JSONL for now (NeMo can process this)
        jsonl_path = output_path + ".jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for tokens in tokens_list:
                # Convert tokens back to text for JSONL format
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                data = {"text": text, "tokens": tokens}
                f.write(json.dumps(data) + "\n")
        
        logging.info(f"Saved {len(tokens_list)} samples to {jsonl_path}")
        
        # Note: For full NeMo integration, you would use:
        # from nemo.collections.nlp.data.language_modeling.megatron.preprocess_data_for_megatron import main as preprocess_main
        # This requires the full NeMo preprocessing pipeline
        
    except Exception as e:
        logging.error(f"Error saving NeMo format: {e}")
        raise


def save_simple_format(tokens_list: List[List[int]], output_path: str, tokenizer):
    """Save data in simple text format."""
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as text files
        text_path = output_path + ".txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            for tokens in tokens_list:
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                f.write(text + "\n\n")
        
        logging.info(f"Saved {len(tokens_list)} samples to {text_path}")
        
    except Exception as e:
        logging.error(f"Error saving simple format: {e}")
        raise


def create_sample_data(output_path: str, num_samples: int = 1000, max_length: int = 512):
    """Create sample data for testing."""
    
    # Sample texts for different domains
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample text for testing the data preparation pipeline.",
        "def hello_world():\n    print('Hello, World!')\n    return 'success'",
        "import torch\nimport torch.nn as nn\n\nclass SimpleModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 1)\n    \n    def forward(self, x):\n        return self.linear(x)",
        "The theory of relativity is a fundamental concept in physics that describes the relationship between space and time.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The internet has revolutionized communication and information sharing across the globe.",
        "Quantum computing represents a paradigm shift in computational capabilities.",
        "Renewable energy sources like solar and wind power are becoming increasingly cost-effective.",
    ]
    
    # Create sample data
    sample_data = []
    for i in range(num_samples):
        # Select a random sample text
        import random
        text = random.choice(sample_texts)
        
        # Add some variation
        text += f" This is sample {i+1} of {num_samples}."
        
        sample_data.append(text)
    
    # Save sample data
    sample_file = os.path.join(output_path, "sample_data.txt")
    os.makedirs(output_path, exist_ok=True)
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        for text in sample_data:
            f.write(text + "\n\n")
    
    logging.info(f"Created {num_samples} sample texts in {sample_file}")
    return sample_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Prepare data for NeMo foundation training")
    
    # Input/Output
    parser.add_argument("--input_path", type=str, help="Path to input text files or directory")
    parser.add_argument("--output_path", type=str, help="Path to save prepared data")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    # Data processing
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3-Coder-480B-A35B-Instruct", help="Tokenizer path")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--min_length", type=int, default=32, help="Minimum sequence length")
    parser.add_argument("--file_extension", type=str, default=".txt", help="File extension to process")
    parser.add_argument("--json_mode", action="store_true", help="Process JSON files")
    parser.add_argument("--text_field", type=str, default="text", help="Field name for text in JSON files")
    
    # Sample data creation
    parser.add_argument("--create_sample", action="store_true", help="Create sample data for testing")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of sample texts to create")
    
    args = parser.parse_args()
    
    if args.create_sample:
        # Create sample data
        if not args.output_path:
            args.output_path = "./sample_data"
        sample_file = create_sample_data(args.output_path, args.num_samples, args.max_length)
        print(f"Sample data created: {sample_file}")
        print("You can now use this file with --input_path to test the data preparation pipeline.")
    else:
        # Prepare data
        if not args.input_path or not args.output_path:
            parser.error("--input_path and --output_path are required when not using --create_sample")
        
        prepare_text_data(
            input_path=args.input_path,
            output_path=args.output_path,
            tokenizer_path=args.tokenizer_path,
            max_length=args.max_length,
            min_length=args.min_length,
            file_extension=args.file_extension,
            json_mode=args.json_mode,
            text_field=args.text_field,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dataset Processor for NeMo ModularModel Pretraining

This script processes Hugging Face pretraining datasets according to the configuration
defined in config.yaml. It handles downloading, preprocessing, and mixing datasets
based on specified percentages.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
import random
import math

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from datasets import load_dataset, Dataset, IterableDataset
    from transformers import AutoTokenizer
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets and transformers not available. Install with: pip install datasets transformers")

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    percentage: float
    max_samples: Optional[int] = None
    text_column: str = "text"
    cache_dir: Optional[str] = None
    subset: Optional[str] = None  # For datasets with specific subsets
    subsets: Optional[List[str]] = None  # For datasets with multiple subsets

@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""
    max_tokens_per_sample: int = 2048
    min_tokens_per_sample: int = 50
    overlap_tokens: int = 0
    shuffle_datasets: bool = True
    seed: int = 42
    save_format: str = "jsonl"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"

class DatasetProcessor:
    """Main class for processing pretraining datasets."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the dataset processor."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.processing_config = self._load_processing_config()
        self.dataset_configs = self._load_dataset_configs()
        self.tokenizer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self._create_directories()
        
        # Set random seed
        random.seed(self.processing_config.seed)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the main configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration."""
        datasets_config = self.config.get("datasets", {})
        processing_config = datasets_config.get("processing", {})
        output_config = datasets_config.get("output", {})
        
        return ProcessingConfig(
            max_tokens_per_sample=processing_config.get("max_tokens_per_sample", 2048),
            min_tokens_per_sample=processing_config.get("min_tokens_per_sample", 50),
            overlap_tokens=processing_config.get("overlap_tokens", 0),
            shuffle_datasets=processing_config.get("shuffle_datasets", True),
            seed=processing_config.get("seed", 42),
            save_format=output_config.get("save_format", "jsonl"),
            processed_data_dir=output_config.get("processed_data_dir", "data/processed"),
            cache_dir=output_config.get("cache_dir", "data/cache")
        )
    
    def _load_dataset_configs(self) -> List[DatasetConfig]:
        """Load dataset configurations."""
        datasets_config = self.config.get("datasets", {})
        pretraining_datasets = datasets_config.get("pretraining_datasets", {})
        output_config = datasets_config.get("output", {})
        
        configs = []
        for name, config_data in pretraining_datasets.items():
            # Handle both old format (percentage as value) and new format (config dict)
            if isinstance(config_data, (int, float)):
                # Old format: "dataset_name": percentage
                percentage = config_data
                subset = None
                subsets = None
            else:
                # New format: "dataset_name": {percentage: X, subset: Y, ...}
                percentage = config_data.get("percentage", 0.0)
                subset = config_data.get("subset")
                subsets = config_data.get("subsets")
            
            configs.append(DatasetConfig(
                name=name,
                percentage=percentage,
                max_samples=None,  # Will be calculated based on total_samples and percentage
                cache_dir=self.processing_config.cache_dir,
                subset=subset,
                subsets=subsets
            ))
        
        # Sort by percentage (descending) for better processing order
        configs.sort(key=lambda x: x.percentage, reverse=True)
        return configs
    
    def _create_directories(self):
        """Create necessary directories."""
        Path(self.processing_config.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processing_config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_tokenizer(self, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"):
        """Load the tokenizer for text processing with caching support."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets and transformers not available")
        
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
    
    def load_dataset(self, dataset_config: DatasetConfig) -> Iterator[Dict[str, Any]]:
        """Load a single dataset."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets and transformers not available")
        
        self.logger.info(f"📥 Loading dataset: {dataset_config.name}")
        
        try:
            # Handle subset specifications
            if dataset_config.subsets:
                # Load multiple subsets and combine them
                all_samples = []
                for subset in dataset_config.subsets:
                    self.logger.info(f"Loading subset '{subset}' from {dataset_config.name}")
                    subset_samples = self._load_single_subset(dataset_config.name, subset, dataset_config.cache_dir)
                    all_samples.extend(subset_samples)
                
                # Process combined samples
                sample_count = 0
                for sample in all_samples:
                    if dataset_config.max_samples and sample_count >= dataset_config.max_samples:
                        break
                    
                    # Extract text from sample
                    text = self._extract_text(sample, dataset_config.text_column)
                    if text:
                        yield {"text": text}
                        sample_count += 1
                        
            elif dataset_config.subset:
                # Load single subset
                self.logger.info(f"Loading subset '{dataset_config.subset}' from {dataset_config.name}")
                samples = self._load_single_subset(dataset_config.name, dataset_config.subset, dataset_config.cache_dir)
                
                sample_count = 0
                for sample in samples:
                    if dataset_config.max_samples and sample_count >= dataset_config.max_samples:
                        break
                    
                    # Extract text from sample
                    text = self._extract_text(sample, dataset_config.text_column)
                    if text:
                        yield {"text": text}
                        sample_count += 1
            else:
                # Load dataset with default or auto-detected config
                dataset = self._load_dataset_with_fallback(dataset_config)
                
                # Process samples
                sample_count = 0
                for sample in dataset:
                    if dataset_config.max_samples and sample_count >= dataset_config.max_samples:
                        break
                    
                    # Extract text from sample
                    text = self._extract_text(sample, dataset_config.text_column)
                    if text:
                        yield {"text": text}
                        sample_count += 1
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset {dataset_config.name}: {e}")
            raise
    
    def _extract_text(self, sample: Dict[str, Any], text_column: str) -> str:
        """Extract text from a dataset sample."""
        if text_column in sample:
            return sample[text_column]
        
        # Try common text column names
        for col in ["text", "content", "body", "article", "code"]:
            if col in sample:
                return sample[col]
        
        # If no text column found, convert the whole sample to string
        return str(sample)
    
    def _load_single_subset(self, dataset_name: str, subset: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
        """Load a single subset from a dataset."""
        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                streaming=True,
                cache_dir=cache_dir
            )
            # Get the first split (usually 'train')
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            
            # Convert to list (for small subsets)
            samples = []
            for sample in dataset:
                samples.append(sample)
                if len(samples) >= 1000:  # Limit to prevent memory issues
                    break
            return samples
            
        except Exception as e:
            self.logger.warning(f"Failed to load subset '{subset}' from {dataset_name}: {e}")
            return []
    
    def _load_dataset_with_fallback(self, dataset_config: DatasetConfig):
        """Load dataset with fallback logic for config detection."""
        try:
            # Try to load with default config first
            dataset = load_dataset(
                dataset_config.name,
                streaming=True,
                cache_dir=dataset_config.cache_dir
            )
            # Get the first split (usually 'train')
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            self.logger.info(f"✅ Loaded {dataset_config.name} (streaming)")
            return dataset
        except Exception as e:
            self.logger.warning(f"Streaming failed for {dataset_config.name}: {e}")
            
            # Try to get available configs and use the first one
            try:
                from datasets import get_dataset_infos
                infos = get_dataset_infos(dataset_config.name)
                configs = list(infos.keys())
                
                if configs:
                    # Use the first available config
                    config_name = configs[0]
                    self.logger.info(f"Using config '{config_name}' for {dataset_config.name}")
                    
                    dataset = load_dataset(
                        dataset_config.name,
                        config_name,
                        streaming=True,
                        cache_dir=dataset_config.cache_dir
                    )
                    split_name = list(dataset.keys())[0]
                    dataset = dataset[split_name]
                    self.logger.info(f"✅ Loaded {dataset_config.name} with config '{config_name}' (streaming)")
                    return dataset
                else:
                    raise ValueError("No configs available")
                    
            except Exception as e2:
                self.logger.warning(f"Config-based loading failed for {dataset_config.name}: {e2}")
                # Fallback to non-streaming
                dataset = load_dataset(
                    dataset_config.name,
                    cache_dir=dataset_config.cache_dir
                )
                split_name = list(dataset.keys())[0]
                dataset = dataset[split_name]
                self.logger.info(f"✅ Loaded {dataset_config.name} (non-streaming)")
                return dataset
    
    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """Process text into training samples with smart chunking strategy."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < self.processing_config.min_tokens_per_sample:
            return []
        
        # If text is shorter than max_tokens, return as single sample
        max_tokens = self.processing_config.max_tokens_per_sample
        if len(tokens) <= max_tokens:
            chunk_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return [{
                "input_ids": tokens,
                "text": chunk_text,
                "length": len(tokens)
            }]
        
        # Smart chunking with overlap
        samples = []
        overlap = self.processing_config.overlap_tokens
        step_size = max_tokens - overlap
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + max_tokens, len(tokens))
            
            # Try to find a good sentence boundary within the chunk
            if end_idx < len(tokens):
                # Look for sentence boundaries in the last 200 tokens of the chunk
                search_start = max(start_idx, end_idx - 200)
                best_end = self._find_sentence_boundary_in_range(tokens, search_start, end_idx)
                if best_end and best_end > start_idx + self.processing_config.min_tokens_per_sample:
                    end_idx = best_end
            
            chunk = tokens[start_idx:end_idx]
            
            if len(chunk) >= self.processing_config.min_tokens_per_sample:
                # Decode back to text
                chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                samples.append({
                    "input_ids": chunk,
                    "text": chunk_text,
                    "length": len(chunk)
                })
            
            # Move start position with overlap
            start_idx = max(start_idx + step_size, end_idx - overlap)
            
            # Prevent infinite loop
            if start_idx >= len(tokens):
                break
        
        return samples
    
    def _find_sentence_boundary_in_range(self, tokens: List[int], start_idx: int, end_idx: int) -> int:
        """Find a sentence boundary within a specific range of tokens."""
        # Decode the range of tokens to find sentence boundaries
        range_tokens = tokens[start_idx:end_idx]
        text = self.tokenizer.decode(range_tokens, skip_special_tokens=True)
        
        # Look for sentence endings from the end backwards
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ['.', '!', '?', '\n']:
                # Find the corresponding token position
                char_pos = i
                token_pos = self._find_token_position_for_char_in_range(range_tokens, char_pos)
                if token_pos is not None:
                    # Convert back to absolute token position
                    absolute_pos = start_idx + token_pos
                    # Make sure it's not too close to the start
                    if absolute_pos - start_idx >= self.processing_config.min_tokens_per_sample:
                        return absolute_pos
        
        return None
    
    def _find_token_position_for_char_in_range(self, tokens: List[int], char_pos: int) -> int:
        """Find the token position corresponding to a character position within a token range."""
        current_pos = 0
        for i, token in enumerate(tokens):
            token_text = self.tokenizer.decode([token], skip_special_tokens=True)
            token_len = len(token_text)
            
            if current_pos <= char_pos < current_pos + token_len:
                return i
            current_pos += token_len
        
        return None
    
    def mix_datasets(self, total_samples: int) -> Iterator[Dict[str, Any]]:
        """Mix datasets according to their percentages."""
        self.logger.info(f"🔄 Mixing datasets for {total_samples} total samples")
        
        # Calculate samples per dataset
        dataset_samples = {}
        for config in self.dataset_configs:
            samples = int(total_samples * config.percentage / 100.0)
            dataset_samples[config.name] = samples
            self.logger.info(f"  {config.name}: {samples} samples ({config.percentage}%)")
        
        # Load and process each dataset
        all_samples = []
        for config in self.dataset_configs:
            target_samples = dataset_samples[config.name]
            if target_samples == 0:
                continue
            
            self.logger.info(f"📥 Processing {config.name} for {target_samples} samples")
            dataset_samples_list = []
            
            for sample in self.load_dataset(config):
                processed_samples = self.process_text(sample["text"])
                dataset_samples_list.extend(processed_samples)
                
                if len(dataset_samples_list) >= target_samples:
                    break
            
            # Take only the required number of samples
            dataset_samples_list = dataset_samples_list[:target_samples]
            all_samples.extend(dataset_samples_list)
            self.logger.info(f"✅ Collected {len(dataset_samples_list)} samples from {config.name}")
        
        # Shuffle if requested
        if self.processing_config.shuffle_datasets:
            self.logger.info("🔀 Shuffling mixed datasets")
            random.shuffle(all_samples)
        
        # Yield samples
        for i, sample in enumerate(all_samples):
            sample["global_id"] = i
            yield sample
    
    def save_processed_data(self, samples: Iterator[Dict[str, Any]], output_path: str):
        """Save processed samples to file."""
        self.logger.info(f"💾 Saving processed data to: {output_path}")
        
        if self.processing_config.save_format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        elif self.processing_config.save_format == "json":
            samples_list = list(samples)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples_list, f, ensure_ascii=False, indent=2)
        elif self.processing_config.save_format == "hf":
            # Save as HuggingFace dataset format
            samples_list = list(samples)
            hf_dataset = Dataset.from_list(samples_list)
            hf_dataset.save_to_disk(output_path)
            self.logger.info(f"✅ Saved HuggingFace dataset with {len(hf_dataset)} samples to: {output_path}")
        else:
            raise ValueError(f"Unsupported save format: {self.processing_config.save_format}")
        
        self.logger.info(f"✅ Saved processed data to: {output_path}")
    
    def process_all_datasets(self, total_samples: int = 10000, output_filename: str = "mixed_pretraining_data"):
        """Process all datasets and create mixed training data."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets and transformers not available")
        
        self.logger.info("🚀 Starting dataset processing")
        self.logger.info(f"📊 Target samples: {total_samples}")
        self.logger.info(f"📁 Output directory: {self.processing_config.processed_data_dir}")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Mix datasets
        mixed_samples = self.mix_datasets(total_samples)
        
        # Save processed data
        output_path = Path(self.processing_config.processed_data_dir) / f"{output_filename}.{self.processing_config.save_format}"
        self.save_processed_data(mixed_samples, str(output_path))
        
        self.logger.info("✅ Dataset processing complete!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process pretraining datasets")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--total_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--output_filename", type=str, default="mixed_pretraining_data",
                       help="Output filename (without extension)")
    parser.add_argument("--tokenizer_path", type=str, 
                       default="tokenizers/qwen3-coder-30b-a3b-instruct-custom",
                       help="Path to tokenizer")
    
    args = parser.parse_args()
    
    if not DATASETS_AVAILABLE:
        print("❌ Required packages not available. Install with:")
        print("pip install datasets transformers")
        return
    
    # Create processor
    processor = DatasetProcessor(args.config)
    
    # Process datasets
    processor.process_all_datasets(
        total_samples=args.total_samples,
        output_filename=args.output_filename
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HuggingFace Dataset Loader for NeMo ModularModel Training

This module provides functionality to load and process HuggingFace datasets
according to the configuration defined in config.yaml.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset as TorchDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceDatasetLoader:
    """
    Loader for HuggingFace datasets with percentage-based allocation.
    
    This class handles loading multiple HuggingFace datasets according to
    the configuration in config.yaml and creates a unified dataset for training.
    """
    
    def __init__(self, config_path: str, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"):
        """
        Initialize the HuggingFace dataset loader.
        
        Args:
            config_path: Path to the config.yaml file
            tokenizer_path: Path to the tokenizer
        """
        self.config_path = Path(config_path)
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.datasets_config = None
        self.loaded_datasets = {}
        
        # Load configuration
        self._load_config()
        
        # Load tokenizer
        self._load_tokenizer()
    
    def _load_config(self):
        """Load the dataset configuration from config.yaml."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.datasets_config = config.get("datasets", {}).get("pretraining_datasets", {})
            logger.info(f"Loaded dataset configuration with {len(self.datasets_config)} datasets")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            # Try local tokenizer first
            if os.path.exists(self.tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                logger.info(f"Loaded local tokenizer from: {self.tokenizer_path}")
            else:
                # Fallback to HuggingFace Hub
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-480B-A35B-Instruct")
                logger.info(f"Loaded tokenizer from HuggingFace Hub: Qwen/Qwen3-Coder-480B-A35B-Instruct")
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_dataset(self, dataset_name: str, subset: Optional[str] = None, 
                    subsets: Optional[List[str]] = None, max_samples: Optional[int] = None) -> Dataset:
        """
        Load a single HuggingFace dataset.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            subset: Single subset to load
            subsets: List of subsets to load
            max_samples: Maximum number of samples to load
            
        Returns:
            Loaded dataset
        """
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            
            if subset:
                logger.info(f"Loading subset: {subset}")
                dataset = load_dataset(dataset_name, subset, split="train", streaming=False)
            elif subsets:
                logger.info(f"Loading subsets: {subsets}")
                # Load multiple subsets and concatenate
                subset_datasets = []
                for sub in subsets:
                    sub_dataset = load_dataset(dataset_name, sub, split="train", streaming=False)
                    subset_datasets.append(sub_dataset)
                dataset = concatenate_datasets(subset_datasets)
            else:
                dataset = load_dataset(dataset_name, split="train", streaming=False)
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Limiting dataset to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def process_dataset(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """
        Process a dataset to extract text content.
        
        Args:
            dataset: Raw dataset
            text_column: Column name containing text content
            
        Returns:
            Processed dataset with text content
        """
        try:
            # Check if the dataset has the expected text column
            if text_column not in dataset.column_names:
                # Try common text column names
                possible_columns = ["text", "content", "body", "article", "passage", "document"]
                for col in possible_columns:
                    if col in dataset.column_names:
                        text_column = col
                        logger.info(f"Using column '{col}' for text content")
                        break
                else:
                    logger.warning(f"No text column found. Available columns: {dataset.column_names}")
                    # Use the first column as text
                    text_column = dataset.column_names[0]
                    logger.info(f"Using first column '{text_column}' as text content")
            
            # Extract text content
            def extract_text(example):
                text = example[text_column]
                if isinstance(text, list):
                    # If it's a list, join the elements
                    text = " ".join(str(item) for item in text)
                return {"text": str(text)}
            
            processed_dataset = dataset.map(extract_text, remove_columns=dataset.column_names)
            
            # Filter out empty or very short texts
            processed_dataset = processed_dataset.filter(lambda x: len(x["text"].strip()) > 10)
            
            logger.info(f"Processed dataset: {len(processed_dataset)} samples")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            raise
    
    def load_all_datasets(self, max_samples_per_dataset: Optional[int] = None) -> Dataset:
        """
        Load all datasets according to the configuration.
        
        Args:
            max_samples_per_dataset: Maximum samples per dataset (for testing)
            
        Returns:
            Combined dataset
        """
        try:
            all_datasets = []
            total_percentage = 0
            
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                subset = config.get("subset")
                subsets = config.get("subsets")
                
                logger.info(f"Loading {dataset_name} ({percentage}%)")
                
                # Load the dataset
                dataset = self.load_dataset(
                    dataset_name=dataset_name,
                    subset=subset,
                    subsets=subsets,
                    max_samples=max_samples_per_dataset
                )
                
                # Process the dataset
                processed_dataset = self.process_dataset(dataset)
                
                # Calculate number of samples based on percentage
                if max_samples_per_dataset:
                    # For testing, use a smaller number
                    num_samples = min(max_samples_per_dataset, len(processed_dataset))
                else:
                    # For production, use percentage (this is a simplified approach)
                    # In practice, you might want to implement more sophisticated sampling
                    num_samples = min(int(len(processed_dataset) * percentage / 100), len(processed_dataset))
                
                if num_samples > 0:
                    sampled_dataset = processed_dataset.select(range(num_samples))
                    all_datasets.append(sampled_dataset)
                    total_percentage += percentage
                    logger.info(f"Added {num_samples} samples from {dataset_name}")
            
            if not all_datasets:
                raise ValueError("No datasets were loaded successfully")
            
            # Combine all datasets
            combined_dataset = concatenate_datasets(all_datasets)
            logger.info(f"Combined dataset: {len(combined_dataset)} total samples")
            logger.info(f"Total percentage: {total_percentage}%")
            
            return combined_dataset
            
        except Exception as e:
            logger.error(f"Failed to load all datasets: {e}")
            raise
    
    def create_training_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create training data in the format expected by the training pipeline.
        
        Args:
            max_samples: Maximum number of samples to create
            
        Returns:
            List of training samples
        """
        try:
            # Load all datasets
            combined_dataset = self.load_all_datasets(max_samples_per_dataset=max_samples)
            
            # Convert to list of dictionaries
            training_data = []
            for i, example in enumerate(combined_dataset):
                if max_samples and i >= max_samples:
                    break
                    
                training_data.append({
                    "text": example["text"],
                    "id": i
                })
            
            logger.info(f"Created {len(training_data)} training samples")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to create training data: {e}")
            raise


class HuggingFaceDatasetWrapper(TorchDataset):
    """
    PyTorch Dataset wrapper for HuggingFace datasets.
    """
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        """
        Initialize the dataset wrapper.
        
        Args:
            data: List of training samples
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single training sample."""
        sample = self.data[idx]
        text = sample["text"]
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # Create labels for next-token prediction
        labels = []
        for i in range(len(tokens)):
            if i == len(tokens) - 1:
                labels.append(-100)  # Last position has no target
            elif tokens[i] == self.tokenizer.pad_token_id:
                labels.append(-100)  # Padding position has no target
            elif i + 1 < len(tokens) and tokens[i + 1] == self.tokenizer.pad_token_id:
                labels.append(-100)  # Next token is padding
            else:
                labels.append(tokens[i + 1])  # Target is the next token
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def load_huggingface_datasets(config_path: str, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
                             max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to load HuggingFace datasets.
    
    Args:
        config_path: Path to config.yaml
        tokenizer_path: Path to tokenizer
        max_samples: Maximum number of samples to load
        
    Returns:
        List of training samples
    """
    loader = HuggingFaceDatasetLoader(config_path, tokenizer_path)
    return loader.create_training_data(max_samples)


if __name__ == "__main__":
    # Test the dataset loader
    config_path = "configs/config.yaml"
    tokenizer_path = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
    
    try:
        print("Testing HuggingFace dataset loader...")
        loader = HuggingFaceDatasetLoader(config_path, tokenizer_path)
        
        # Load a small sample for testing
        training_data = loader.create_training_data(max_samples=100)
        print(f"Loaded {len(training_data)} training samples")
        
        # Test the wrapper
        wrapper = HuggingFaceDatasetWrapper(training_data, loader.tokenizer, max_length=512)
        sample = wrapper[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

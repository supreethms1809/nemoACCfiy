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
import os
import boto3
from smart_open import open
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
    
    def __init__(self, config_path: str, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom", stage: str = "stage1"):
        """
        Initialize the HuggingFace dataset loader.
        
        Args:
            config_path: Path to the config.yaml file
            tokenizer_path: Path to the tokenizer
            stage: Training stage ("stage1", "stage2", "stage3")
        """
        self.config_path = Path(config_path)
        self.tokenizer_path = tokenizer_path
        self.stage = stage
        self.tokenizer = None
        self.datasets_config = None
        self.global_settings = None
        self.stage_processing = None
        self.loaded_datasets = {}
        self.s3_client = None
        
        # Initialize S3 client for the-stack dataset content download
        self._init_s3_client()
        
        # Load configuration
        self._load_config()
        
        # Load tokenizer
        self._load_tokenizer()
    
    def _init_s3_client(self):
        """Initialize S3 client for downloading the-stack dataset content."""
        try:
            # Check if AWS credentials are available
            if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
                session = boto3.Session(
                    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
                )
                self.s3_client = session.client("s3")
                logger.info("S3 client initialized for the-stack dataset content download")
            else:
                logger.warning("AWS credentials not found. the-stack dataset will use placeholder content.")
                self.s3_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}. the-stack dataset will use placeholder content.")
            self.s3_client = None
    
    def _check_local_stack_dataset(self, dataset_name: str) -> Optional[str]:
        """
        Check if a local downloaded the-stack dataset exists.
        
        Args:
            dataset_name: Name of the dataset to check
            
        Returns:
            Path to local dataset if found, None otherwise
        """
        if dataset_name != "bigcode/the-stack-v2-dedup":
            return None
        
        # Check common local dataset locations (prioritize HuggingFace cache)
        possible_paths = [
            # HuggingFace cache locations (preferred) - D: drive
            "/mnt/d/huggingface_cache/datasets/bigcode___the-stack-v2-dedup-content/stack_content_dataset",
            os.path.expanduser("~/.cache/huggingface/datasets/bigcode___the-stack-v2-dedup-content/stack_content_dataset"),
            os.path.expanduser("~/.cache/huggingface/bigcode___the-stack-v2-dedup-content/stack_content_dataset"),
            # Local project directories (fallback)
            "data/stack_content/stack_content_dataset",
            "data/stack_content_test/stack_content_dataset", 
            "data/stack_large/stack_content_dataset",
            "data/stack_sample/stack_content_dataset"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ Found local the-stack dataset at: {path}")
                return path
        
        logger.info("ℹ️ No local the-stack dataset found, will use S3 download or placeholder")
        return None
    
    def _download_stack_content(self, blob_id: str, src_encoding: str) -> str:
        """
        Download content from the-stack dataset S3 bucket.
        
        Args:
            blob_id: Software Heritage blob ID
            src_encoding: Source encoding of the file
            
        Returns:
            Downloaded content as string
        """
        if not self.s3_client:
            # Return placeholder if S3 client is not available
            return f"# Placeholder content for blob_id: {blob_id}\n# S3 client not available - AWS credentials required"
        
        try:
            s3_url = f"s3://softwareheritage/content/{blob_id}"
            
            with open(s3_url, "rb", compression=".gz", transport_params={"client": self.s3_client}) as fin:
                content = fin.read().decode(src_encoding)
            
            return content
        except Exception as e:
            logger.warning(f"Failed to download content for blob_id {blob_id}: {e}")
            return f"# Error downloading content for blob_id: {blob_id}\n# Error: {str(e)}"
    
    def _load_config(self):
        """Load the dataset configuration from config.yaml."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract global settings
            self.global_settings = config.get('datasets', {}).get('global_settings', {})
            
            # Extract stage-specific configuration
            stage_key = f"{self.stage}_datasets"
            stage_config = config.get('datasets', {}).get(stage_key, {})
            
            if not stage_config:
                logger.warning(f"No {stage_key} configuration found in config")
                return
            
            # Get the appropriate dataset type based on stage
            if self.stage == "stage1":
                self.datasets_config = stage_config.get('pretraining_datasets', {})
            elif self.stage == "stage2":
                self.datasets_config = stage_config.get('finetuning_datasets', {})
            elif self.stage == "stage3":
                self.datasets_config = stage_config.get('instruction_datasets', {})
            else:
                logger.error(f"Unknown stage: {self.stage}")
                return
            
            # Get stage-specific processing settings
            self.stage_processing = stage_config.get('processing', {})
            
            if not self.datasets_config:
                logger.warning(f"No datasets found for {self.stage}")
                return
            
            logger.info(f"Loaded {self.stage} dataset configuration with {len(self.datasets_config)} datasets")
            
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
            
            # Check for local downloaded the-stack dataset first
            local_dataset_path = self._check_local_stack_dataset(dataset_name)
            if local_dataset_path:
                logger.info(f"🔄 Using local downloaded dataset from: {local_dataset_path}")
                try:
                    local_dataset = Dataset.load_from_disk(local_dataset_path)
                    if max_samples and len(local_dataset) > max_samples:
                        local_dataset = local_dataset.select(range(max_samples))
                    logger.info(f"✅ Loaded {len(local_dataset)} samples from local dataset")
                    return local_dataset
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load local dataset: {e}. Falling back to remote loading.")
            
            # Get cache directory from global settings
            cache_dir = self.global_settings.get('cache_dir', '~/.cache/huggingface')
            
            if subset:
                logger.info(f"Loading subset: {subset}")
                # Try streaming first to avoid rate limits
                try:
                    dataset = load_dataset(dataset_name, subset, split="train", streaming=True, cache_dir=cache_dir)
                    # Convert streaming dataset to regular dataset with limited samples
                    if max_samples:
                        samples = []
                        for i, sample in enumerate(dataset):
                            if i >= max_samples:
                                break
                            samples.append(sample)
                        # Create a new dataset from the samples
                        from datasets import Dataset as HFDataset
                        dataset = HFDataset.from_list(samples)
                    else:
                        # Take first 1000 samples by default for streaming
                        samples = []
                        for i, sample in enumerate(dataset):
                            if i >= 1000:
                                break
                            samples.append(sample)
                        from datasets import Dataset as HFDataset
                        dataset = HFDataset.from_list(samples)
                except Exception as stream_error:
                    logger.warning(f"Streaming failed for {dataset_name}, trying regular load: {stream_error}")
                    dataset = load_dataset(dataset_name, subset, split="train", streaming=False, cache_dir=cache_dir)
            elif subsets:
                logger.info(f"Loading subsets: {subsets}")
                # Load multiple subsets and concatenate
                subset_datasets = []
                for sub in subsets:
                    try:
                        sub_dataset = load_dataset(dataset_name, sub, split="train", streaming=False, cache_dir=cache_dir)
                        subset_datasets.append(sub_dataset)
                    except Exception as sub_error:
                        logger.warning(f"Failed to load subset {sub} of {dataset_name}: {sub_error}")
                        continue
                
                if not subset_datasets:
                    raise ValueError(f"Failed to load any subsets from {dataset_name}")
                dataset = concatenate_datasets(subset_datasets)
            else:
                # Try streaming first, then fallback to non-streaming
                try:
                    logger.info(f"Loading dataset without subset (streaming)")
                    dataset = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir)
                    # Convert streaming dataset to regular dataset with limited samples
                    if max_samples:
                        samples = []
                        for i, sample in enumerate(dataset):
                            if i >= max_samples:
                                break
                            samples.append(sample)
                        # Create a new dataset from the samples
                        from datasets import Dataset as HFDataset
                        dataset = HFDataset.from_list(samples)
                    else:
                        # Take first 1000 samples by default for streaming
                        samples = []
                        for i, sample in enumerate(dataset):
                            if i >= 1000:
                                break
                            samples.append(sample)
                        from datasets import Dataset as HFDataset
                        dataset = HFDataset.from_list(samples)
                except Exception as stream_error:
                    logger.warning(f"Streaming failed for {dataset_name}, trying regular load: {stream_error}")
                    dataset = load_dataset(dataset_name, split="train", streaming=False, cache_dir=cache_dir)
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Limiting dataset to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def process_dataset(self, dataset: Dataset, text_column: str = "text", task_type: str = "general") -> Dataset:
        """
        Process a dataset to extract text content with stage-specific processing.
        
        Args:
            dataset: Raw dataset
            text_column: Column name containing text content
            task_type: Type of task (general, code_completion, instruction_following, etc.)
            
        Returns:
            Processed dataset with text content
        """
        try:
            # Check if the dataset has the expected text column
            if text_column not in dataset.column_names:
                # Try common text column names
                possible_columns = ["text", "content", "body", "article", "passage", "document", "code", "instruction", "response", "src", "source"]
                for col in possible_columns:
                    if col in dataset.column_names:
                        text_column = col
                        logger.info(f"Using column '{col}' for text content")
                        break
                else:
                    logger.warning(f"No text column found. Available columns: {dataset.column_names}")
                    # For the-stack datasets, we need to download content from S3
                    if "blob_id" in dataset.column_names and "src_encoding" in dataset.column_names:
                        logger.info("Detected the-stack dataset format. Content will be downloaded from S3.")
                        # We'll handle this in the extract_text function
                        text_column = "blob_id"  # Use blob_id as placeholder
                    else:
                        logger.error(f"Cannot find text content in dataset. Skipping this dataset.")
                        return dataset.filter(lambda x: False)  # Return empty dataset
            
            # Extract text content with stage-specific processing
            def extract_text(example):
                # Handle the-stack dataset format
                if text_column == "blob_id" and "blob_id" in example and "src_encoding" in example:
                    # For the-stack datasets, download actual content from S3
                    blob_id = example["blob_id"]
                    src_encoding = example.get("src_encoding", "utf-8")
                    
                    # Download actual content from S3
                    text = self._download_stack_content(blob_id, src_encoding)
                    
                    logger.info(f"Downloaded content for the-stack dataset. Blob ID: {blob_id}, Length: {len(text)} chars")
                else:
                    text = example[text_column]
                    if isinstance(text, list):
                        # If it's a list, join the elements
                        text = " ".join(str(item) for item in text)
                    
                    # Stage-specific text processing
                    if self.stage == "stage3" and task_type in ["instruction_following", "code_instruction", "math_instruction"]:
                        # For instruction tuning, we might need to format the text differently
                        # This is a placeholder for more sophisticated instruction formatting
                        text = str(text)
                    else:
                        text = str(text)
                
                return {"text": text}
            
            processed_dataset = dataset.map(extract_text, remove_columns=dataset.column_names)
            
            # Get stage-specific filtering criteria
            min_length = self.stage_processing.get('min_tokens_per_sample', 50)
            if self.stage == "stage2":
                min_length = max(min_length, 100)  # Longer minimum for fine-tuning
            elif self.stage == "stage3":
                min_length = max(min_length, 200)  # Even longer for instruction tuning
            
            # Filter out empty or very short texts
            processed_dataset = processed_dataset.filter(lambda x: len(x["text"].strip()) > min_length)
            
            logger.info(f"Processed {self.stage} dataset for {task_type}: {len(processed_dataset)} samples")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Failed to process dataset for {self.stage}: {e}")
            raise
    
    def load_all_datasets(self, max_samples_per_dataset: Optional[int] = None) -> Dataset:
        """
        Load all datasets according to the stage-specific configuration.
        
        Args:
            max_samples_per_dataset: Maximum samples per dataset (for testing)
            
        Returns:
            Combined dataset
        """
        try:
            all_datasets = []
            total_percentage = 0
            
            failed_datasets = []
            
            # Get max samples from stage processing config if not provided
            if max_samples_per_dataset is None:
                max_samples_per_dataset = self.stage_processing.get('max_samples_per_dataset')
            
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                subset = config.get("subset")
                subsets = config.get("subsets")
                task_type = config.get("task_type", "general")
                
                logger.info(f"Loading {dataset_name} ({percentage}%) for {self.stage} - {task_type}")
                
                try:
                    # Load the dataset
                    dataset = self.load_dataset(
                        dataset_name=dataset_name,
                        subset=subset,
                        subsets=subsets,
                        max_samples=max_samples_per_dataset
                    )
                    
                    # Process the dataset with stage-specific settings
                    processed_dataset = self.process_dataset(dataset, task_type=task_type)
                    
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
                        logger.info(f"Added {num_samples} samples from {dataset_name} for {task_type}")
                    else:
                        logger.warning(f"No samples selected from {dataset_name}")
                        
                except Exception as dataset_error:
                    logger.error(f"Failed to load dataset {dataset_name}: {dataset_error}")
                    failed_datasets.append(dataset_name)
                    continue
            
            if not all_datasets:
                raise ValueError(f"No datasets were loaded successfully for {self.stage}. Failed datasets: {failed_datasets}")
            
            if failed_datasets:
                logger.warning(f"Some datasets failed to load for {self.stage}: {failed_datasets}")
            
            # Combine all datasets
            combined_dataset = concatenate_datasets(all_datasets)
            logger.info(f"Combined {self.stage} dataset: {len(combined_dataset)} total samples")
            logger.info(f"Total percentage: {total_percentage}%")
            
            return combined_dataset
            
        except Exception as e:
            logger.error(f"Failed to load all datasets for {self.stage}: {e}")
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

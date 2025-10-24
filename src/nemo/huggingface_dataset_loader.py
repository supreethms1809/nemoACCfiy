#!/usr/bin/env python3
"""
HuggingFace Dataset Loader for NeMo ModularModel Training

This module provides functionality to load and process HuggingFace datasets
according to the configuration defined in config.yaml.
"""

import os
import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Iterator
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


class BufferedStreamingDataset:
    """
    High-performance buffered streaming dataset for GH200 nodes with large memory.
    
    This class implements a large memory buffer that pre-loads samples from streaming
    datasets, taking advantage of the 400+ GB RAM available on GH200 nodes.
    
    Single-threaded implementation that's thread-safe and reliable.
    """
    
    def __init__(self, 
                 dataset,  # HuggingFace streaming dataset
                 buffer_size: int = 1000000,
                 refill_threshold: float = 0.2):
        """
        Initialize the buffered streaming dataset.
        
        Args:
            dataset: HuggingFace streaming dataset
            buffer_size: Maximum number of samples to keep in buffer
            refill_threshold: Refill buffer when this fraction is consumed
        """
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.refill_threshold = refill_threshold
        
        # Buffer management
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.dataset_iterator = iter(dataset)
        self.total_loaded = 0
        self.total_consumed = 0
        self.exhausted = False
        
        # Pre-fill buffer
        self._refill_buffer()
        
        logger.info(f"Initialized BufferedStreamingDataset with buffer_size={buffer_size}, "
                   f"refill_threshold={refill_threshold}")
    
    def _refill_buffer(self):
        """Refill the buffer with new samples."""
        with self.buffer_lock:
            target_size = int(self.buffer_size * (1 - self.refill_threshold))
            
            while len(self.buffer) < target_size and not self.exhausted:
                try:
                    sample = next(self.dataset_iterator)
                    self.buffer.append(sample)
                    self.total_loaded += 1
                    
                    if self.total_loaded % 50000 == 0:
                        logger.info(f"Loaded {self.total_loaded} samples, "
                                   f"Buffer size: {len(self.buffer)}")
                        
                except StopIteration:
                    self.exhausted = True
                    logger.info("Dataset exhausted")
                    break
                except Exception as e:
                    logger.error(f"Error loading sample: {e}")
                    break
    
    def __iter__(self):
        """Iterate over samples from the buffer."""
        return self
    
    def __next__(self):
        """Get next sample from buffer."""
        with self.buffer_lock:
            # Check if we need to refill buffer
            if len(self.buffer) <= self.buffer_size * self.refill_threshold and not self.exhausted:
                # Refill buffer in background (this is still single-threaded but more efficient)
                self._refill_buffer()
            
            # Get sample from buffer
            if self.buffer:
                sample = self.buffer.pop(0)
                self.total_consumed += 1
                
                if self.total_consumed % 50000 == 0:
                    logger.info(f"Consumed {self.total_consumed} samples, "
                               f"Buffer size: {len(self.buffer)}")
                
                return sample
            else:
                raise StopIteration("All samples consumed")
    
    def __len__(self):
        """Return approximate number of samples (not exact for streaming)."""
        return self.total_loaded
    
    def close(self):
        """Clean up resources."""
        logger.info(f"BufferedStreamingDataset closed. Total loaded: {self.total_loaded}, "
                   f"Total consumed: {self.total_consumed}")

class HuggingFaceDatasetLoader:
    """
    Loader for HuggingFace datasets with percentage-based allocation.
    
    This class handles loading multiple HuggingFace datasets according to
    the configuration in config.yaml and creates a unified dataset for training.
    """
    
    def __init__(self, config_path: str, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom", stage: str = "stage1", enable_buffer: bool = None):
        """
        Initialize the HuggingFace dataset loader.
        
        Args:
            config_path: Path to the config.yaml file
            tokenizer_path: Path to the tokenizer
            stage: Training stage ("stage1", "stage2", "stage3")
            enable_buffer: Override buffer setting (None = use config, True/False = override)
        """
        self.config_path = Path(config_path)
        self.tokenizer_path = tokenizer_path
        self.stage = stage
        self.enable_buffer = enable_buffer
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
            
            # Extract stage-specific configuration from training_stages
            training_stages = config.get('training_stages', {})
            stage_config = training_stages.get(self.stage, {})
            
            if not stage_config:
                logger.warning(f"No {self.stage} configuration found in training_stages")
                return
            
            # Get data configuration from the stage
            data_config = stage_config.get('data', {})
            
            if not data_config:
                logger.warning(f"No data configuration found for {self.stage}")
                return
            
            # Extract global settings from data config
            self.global_settings = {
                'use_streaming': data_config.get('use_streaming', True),
                'streaming_fallback': data_config.get('streaming_fallback', True),
                'cache_datasets': data_config.get('cache_datasets', True),
                'processed_data_dir': data_config.get('processed_data_dir', 'data/processed'),
                'cache_dir': data_config.get('cache_dir', '~/.cache/huggingface'),
                'save_format': data_config.get('save_format', 'jsonl'),
                'shuffle_datasets': data_config.get('shuffle_datasets', True),
                'seed': data_config.get('seed', 42),
                # Memory buffer settings for GH200 optimization
                'use_memory_buffer': data_config.get('use_memory_buffer', True),
                'buffer_size_mb': data_config.get('buffer_size_mb', 200000),  # 200GB default for GH200
                'buffer_samples': data_config.get('buffer_samples', 5000000),  # 5M samples
                'buffer_refill_threshold': data_config.get('buffer_refill_threshold', 0.1),  # More aggressive
                'enable_memory_mapping': data_config.get('enable_memory_mapping', True)
            }
            
            # Get the appropriate dataset type based on stage
            if self.stage == "stage1":
                self.datasets_config = data_config.get('pretraining_datasets', {})
            elif self.stage == "stage2":
                self.datasets_config = data_config.get('finetuning_datasets', {})
            elif self.stage == "stage3":
                self.datasets_config = data_config.get('instruction_datasets', {})
            else:
                logger.error(f"Unknown stage: {self.stage}")
                return
            
            # Get stage-specific processing settings
            self.stage_processing = data_config.get('processing', {})
            
            if not self.datasets_config:
                logger.warning(f"No datasets found for {self.stage}")
                return
            
            logger.info(f"Loaded {self.stage} dataset configuration with {len(self.datasets_config)} datasets")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer with caching support."""
        try:
            # Import tokenizer manager
            from src.utils.tokenizer_manager import get_tokenizer_with_caching
            
            # Use the caching system
            logger.info(f"Loading tokenizer with caching support...")
            self.tokenizer = get_tokenizer_with_caching(
                tokenizer_path=self.tokenizer_path,
                custom_tokens=None,  # Use default special tokens
                force_download=False,
                cache_dir="tokenizers"
            )
            logger.info(f"✅ Tokenizer loaded with vocab size: {len(self.tokenizer)}")
                
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
                start_time = time.time()
                # Try streaming first to avoid rate limits
                try:
                    # Try to load non-streaming first (faster for cached datasets)
                    try:
                        dataset = load_dataset(dataset_name, subset, split="train", streaming=False, cache_dir=cache_dir)
                        # If we have a limit, take only what we need
                        if max_samples and len(dataset) > max_samples:
                            # Use random sampling for better data distribution
                            import random
                            indices = random.sample(range(len(dataset)), max_samples)
                            dataset = dataset.select(indices)
                        elif max_samples and len(dataset) <= max_samples:
                            logger.info(f"Dataset has {len(dataset)} samples, using all (requested {max_samples})")
                    except Exception as non_stream_error:
                        logger.warning(f"Non-streaming failed for {dataset_name}, trying streaming: {non_stream_error}")
                        # Fallback to streaming with efficient sampling
                        dataset = load_dataset(dataset_name, subset, split="train", streaming=True, cache_dir=cache_dir)
                        if max_samples:
                            # Use take() method for efficient sampling from streaming dataset
                            dataset = dataset.take(max_samples)
                        else:
                            raise ValueError(f"max_samples must be specified for {dataset_name}. This prevents downloading entire datasets.")
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
                        # This should not happen with our new approach - we always specify max_samples
                        raise ValueError(f"max_samples must be specified for {dataset_name}. This prevents downloading entire datasets.")
                except Exception as stream_error:
                    logger.warning(f"Streaming failed for {dataset_name}, trying regular load: {stream_error}")
                    dataset = load_dataset(dataset_name, split="train", streaming=False, cache_dir=cache_dir)
                    # Limit samples if specified
                    if max_samples and len(dataset) > max_samples:
                        logger.info(f"Limiting dataset to {max_samples} samples")
                        dataset = dataset.select(range(max_samples))
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Limiting dataset to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            end_time = time.time()
            loading_time = end_time - start_time
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name} in {loading_time:.2f} seconds")
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
            def extract_text(examples):
                # Handle batched processing
                texts = []
                for i in range(len(examples[text_column])):
                    example = {key: examples[key][i] for key in examples.keys()}
                    
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
                    
                    texts.append(text)
                
                return {"text": texts}
            
            # Use batched processing for better performance
            processed_dataset = dataset.map(extract_text, remove_columns=dataset.column_names, batched=True, batch_size=1000)
            
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
    
    def _check_preprocessed_dataset(self, stage: str) -> Optional[Dataset]:
        """
        Check if a preprocessed dataset exists and load it.
        
        Args:
            stage: Training stage
            
        Returns:
            Preprocessed dataset if found, None otherwise
        """
        try:
            from datasets import load_from_disk
            
            processed_dir = Path("data") / stage / "processed"
            dataset_path = processed_dir / "combined_dataset"
            
            if dataset_path.exists():
                logger.info(f"📥 Found preprocessed dataset at {dataset_path}")
                dataset = load_from_disk(str(dataset_path))
                logger.info(f"✅ Loaded preprocessed dataset with {len(dataset)} samples")
                return dataset
            else:
                logger.info(f"📭 No preprocessed dataset found at {dataset_path}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load preprocessed dataset: {e}")
            return None

    def load_all_datasets(self, total_samples: Optional[int] = None) -> Dataset:
        """
        Load all datasets according to the stage-specific configuration.
        
        Args:
            total_samples: Total number of samples across all datasets (code calculates per-dataset based on percentages)
            
        Returns:
            Combined dataset
        """
        try:
            # Check if we should use preprocessed datasets
            use_processed = self.stage_processing.get('use_processed_datasets', False)
            
            if use_processed:
                logger.info("🔍 Checking for preprocessed datasets...")
                preprocessed_dataset = self._check_preprocessed_dataset(self.stage)
                if preprocessed_dataset is not None:
                    logger.info("✅ Using preprocessed dataset (much faster!)")
                    return preprocessed_dataset
                else:
                    logger.warning("⚠️ Preprocessed dataset not found, falling back to live loading")
            
            all_datasets = []
            total_percentage = 0
            
            failed_datasets = []
            
            # Get total samples from stage processing config if not provided
            if total_samples is None:
                total_samples = self.stage_processing.get('total_samples')
            
            if total_samples is None:
                raise ValueError(f"total_samples must be specified either as parameter or in stage processing config")
            
            # Calculate samples per dataset based on percentages
            dataset_samples = {}
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                samples = int(total_samples * (percentage / 100.0))
                dataset_samples[dataset_name] = samples
                logger.info(f"Allocated {samples} samples ({percentage}%) for {dataset_name}")
            
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                subset = config.get("subset")
                subsets = config.get("subsets")
                task_type = config.get("task_type", "general")
                target_samples = dataset_samples[dataset_name]
                
                if target_samples == 0:
                    logger.info(f"Skipping {dataset_name} (0 samples allocated)")
                    continue
                
                logger.info(f"Loading {dataset_name} ({percentage}%) for {self.stage} - {task_type} (target: {target_samples} samples)")
                
                try:
                    # Load the dataset with the exact target sample count (efficient streaming)
                    dataset = self.load_dataset(
                        dataset_name=dataset_name,
                        subset=subset,
                        subsets=subsets,
                        max_samples=target_samples  # Load only what we need
                    )
                    
                    # Process the dataset with stage-specific settings (optimized)
                    process_start = time.time()
                    processed_dataset = self.process_dataset(dataset, task_type=task_type)
                    process_time = time.time() - process_start
                    logger.info(f"Processed {dataset_name} in {process_time:.2f} seconds")
                    
                    # The dataset should already have the right number of samples
                    if len(processed_dataset) >= target_samples:
                        # Take exactly the target number of samples
                        sampled_dataset = processed_dataset.select(range(target_samples))
                    else:
                        # Use all available data if less than target
                        sampled_dataset = processed_dataset
                        logger.warning(f"Only {len(processed_dataset)} samples available for {dataset_name}, requested {target_samples}")
                    
                    all_datasets.append(sampled_dataset)
                    total_percentage += percentage
                    logger.info(f"Added {len(sampled_dataset)} samples from {dataset_name} for {task_type}")
                        
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
    
    def create_training_data(self, total_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create training data in the format expected by the training pipeline.
        
        Args:
            total_samples: Total number of samples across all datasets (code calculates per-dataset based on percentages)
            
        Returns:
            List of training samples
        """
        try:
            # Check if we should use streaming approach
            use_streaming = self.global_settings.get('use_streaming', True)
            
            if use_streaming:
                logger.info("Using streaming approach for memory efficiency")
                return self._create_training_data_streaming(total_samples)
            else:
                logger.info("Using batch loading approach")
                return self._create_training_data_batch(total_samples)
            
        except Exception as e:
            logger.error(f"Failed to create training data: {e}")
            raise
    
    def _create_training_data_streaming(self, total_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create training data using buffered streaming approach optimized for GH200.
        
        Args:
            total_samples: Total number of samples across all datasets
            
        Returns:
            List of training samples
        """
        try:
            # Get total samples from stage processing config if not provided
            if total_samples is None:
                total_samples = self.stage_processing.get('total_samples')
            
            if total_samples is None:
                raise ValueError(f"total_samples must be specified either as parameter or in stage processing config")
            
            # Check if we should use memory buffer
            if self.enable_buffer is not None:
                use_memory_buffer = self.enable_buffer
            else:
                use_memory_buffer = self.global_settings.get('use_memory_buffer', False)
            buffer_samples = self.global_settings.get('buffer_samples', 5000000)
            buffer_refill_threshold = self.global_settings.get('buffer_refill_threshold', 0.1)
            
            if use_memory_buffer:
                logger.info(f"🚀 Using GH200-optimized buffered streaming with {buffer_samples} sample buffer")
                return self._create_training_data_buffered_streaming(total_samples, buffer_samples, buffer_refill_threshold)
            else:
                logger.info("📊 Using efficient streaming approach")
                return self._create_training_data_standard_streaming(total_samples)
            
        except Exception as e:
            logger.error(f"Failed to create streaming training data: {e}")
            raise
    
    def _create_training_data_buffered_streaming(self, total_samples: int, buffer_samples: int, refill_threshold: float) -> List[Dict[str, Any]]:
        """
        Create training data using buffered streaming optimized for GH200 nodes.
        
        Args:
            total_samples: Total number of samples across all datasets
            buffer_samples: Number of samples to keep in memory buffer
            refill_threshold: Refill buffer when this fraction is consumed
            
        Returns:
            List of training samples
        """
        try:
            # Calculate samples per dataset based on percentages
            dataset_samples = {}
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                samples = int(total_samples * (percentage / 100.0))
                dataset_samples[dataset_name] = samples
                logger.info(f"Allocated {samples} samples ({percentage}%) for {dataset_name}")
            
            training_data = []
            total_processed = 0
            
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                subset = config.get("subset")
                subsets = config.get("subsets")
                task_type = config.get("task_type", "general")
                target_samples = dataset_samples[dataset_name]
                
                if target_samples == 0:
                    logger.info(f"Skipping {dataset_name} (0 samples allocated)")
                    continue
                
                logger.info(f"🚀 Buffered streaming {dataset_name} ({percentage}%) for {self.stage} - {task_type} (target: {target_samples} samples)")
                
                try:
                    # Load streaming dataset
                    cache_dir = self.global_settings.get('cache_dir', '~/.cache/huggingface')
                    
                    if subset:
                        dataset = load_dataset(dataset_name, subset, split="train", streaming=True, cache_dir=cache_dir)
                    elif subsets:
                        # For multiple subsets, we'll process them one by one
                        for sub in subsets:
                            sub_dataset = load_dataset(dataset_name, sub, split="train", streaming=True, cache_dir=cache_dir)
                            sub_target = target_samples // len(subsets)
                            self._process_buffered_streaming_dataset(sub_dataset, training_data, sub_target, task_type, total_processed, buffer_samples, refill_threshold)
                            total_processed += sub_target
                        continue
                    else:
                        dataset = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir)
                    
                    # Process streaming dataset with buffer
                    self._process_buffered_streaming_dataset(dataset, training_data, target_samples, task_type, total_processed, buffer_samples, refill_threshold)
                    total_processed += target_samples
                    
                except Exception as dataset_error:
                    logger.error(f"Failed to load dataset {dataset_name}: {dataset_error}")
                    continue
            
            logger.info(f"✅ Created {len(training_data)} training samples using GH200-optimized buffered streaming")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to create buffered streaming training data: {e}")
            raise
    
    def _create_training_data_standard_streaming(self, total_samples: int) -> List[Dict[str, Any]]:
        """
        Create training data using standard streaming approach.
        
        Args:
            total_samples: Total number of samples across all datasets
            
        Returns:
            List of training samples
        """
        try:
            # Calculate samples per dataset based on percentages
            dataset_samples = {}
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                samples = int(total_samples * (percentage / 100.0))
                dataset_samples[dataset_name] = samples
                logger.info(f"Allocated {samples} samples ({percentage}%) for {dataset_name}")
            
            training_data = []
            total_processed = 0
            
            for dataset_name, config in self.datasets_config.items():
                percentage = config.get("percentage", 0)
                subset = config.get("subset")
                subsets = config.get("subsets")
                task_type = config.get("task_type", "general")
                target_samples = dataset_samples[dataset_name]
                
                if target_samples == 0:
                    logger.info(f"Skipping {dataset_name} (0 samples allocated)")
                    continue
                
                logger.info(f"Streaming {dataset_name} ({percentage}%) for {self.stage} - {task_type} (target: {target_samples} samples)")
                
                try:
                    # Load streaming dataset
                    cache_dir = self.global_settings.get('cache_dir', '~/.cache/huggingface')
                    
                    if subset:
                        dataset = load_dataset(dataset_name, subset, split="train", streaming=True, cache_dir=cache_dir)
                    elif subsets:
                        # For multiple subsets, we'll process them one by one
                        for sub in subsets:
                            sub_dataset = load_dataset(dataset_name, sub, split="train", streaming=True, cache_dir=cache_dir)
                            sub_target = target_samples // len(subsets)
                            self._process_streaming_dataset(sub_dataset, training_data, sub_target, task_type, total_processed)
                            total_processed += sub_target
                        continue
                    else:
                        dataset = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir)
                    
                    # Process streaming dataset
                    self._process_streaming_dataset(dataset, training_data, target_samples, task_type, total_processed)
                    total_processed += target_samples
                    
                except Exception as dataset_error:
                    logger.error(f"Failed to load dataset {dataset_name}: {dataset_error}")
                    continue
            
            logger.info(f"Created {len(training_data)} training samples using standard streaming approach")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to create standard streaming training data: {e}")
            raise
    
    def _process_streaming_dataset(self, dataset, training_data: List, target_samples: int, task_type: str, start_id: int):
        """
        Process a streaming dataset and add samples to training_data.
        
        Args:
            dataset: Streaming dataset
            training_data: List to append samples to
            target_samples: Number of samples to process
            task_type: Type of task for processing
            start_id: Starting ID for samples
        """
        processed_count = 0
        
        for i, example in enumerate(dataset):
            if processed_count >= target_samples:
                break
            
            try:
                # Process the sample
                processed_sample = self._process_single_sample(example, task_type)
                if processed_sample:
                    training_data.append({
                        "text": processed_sample,
                        "id": start_id + processed_count
                    })
                    processed_count += 1
                
                # Log progress (less frequent for better performance)
                if processed_count % 50000 == 0:
                    logger.info(f"Processed {processed_count}/{target_samples} samples...")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info(f"Processed {processed_count} samples from streaming dataset")
    
    def _process_buffered_streaming_dataset(self, dataset, training_data: List, target_samples: int, task_type: str, start_id: int, buffer_samples: int, refill_threshold: float):
        """
        Process a streaming dataset using buffered approach optimized for GH200.
        
        Args:
            dataset: Streaming dataset
            training_data: List to append samples to
            target_samples: Number of samples to process
            task_type: Type of task for processing
            start_id: Starting ID for samples
            buffer_samples: Number of samples to keep in buffer
            refill_threshold: Refill buffer when this fraction is consumed
        """
        try:
            # Create buffered streaming dataset
            buffered_dataset = BufferedStreamingDataset(
                dataset=dataset,  # Pass the dataset, not iterator
                buffer_size=buffer_samples,
                refill_threshold=refill_threshold
            )
            
            processed_count = 0
            
            for sample in buffered_dataset:
                if processed_count >= target_samples:
                    break
                
                try:
                    # Process the sample
                    processed_sample = self._process_single_sample(sample, task_type)
                    if processed_sample:
                        training_data.append({
                            "text": processed_sample,
                            "id": start_id + processed_count
                        })
                        processed_count += 1
                    
                    # Log progress (less frequent for better performance)
                    if processed_count % 50000 == 0:
                        logger.info(f"🚀 Buffered processing: {processed_count}/{target_samples} samples...")
                        
                except Exception as e:
                    logger.warning(f"Failed to process sample: {e}")
                    continue
            
            # Clean up buffered dataset
            buffered_dataset.close()
            
            logger.info(f"✅ Buffered processed {processed_count} samples from streaming dataset")
            
        except Exception as e:
            logger.error(f"Failed to process buffered streaming dataset: {e}")
            raise
    
    def _process_single_sample(self, example: Dict, task_type: str) -> Optional[str]:
        """
        Process a single sample from a dataset.
        
        Args:
            example: Single sample from dataset
            task_type: Type of task
            
        Returns:
            Processed text or None if processing failed
        """
        try:
            # Find text column
            text_column = None
            possible_columns = ["text", "content", "body", "article", "passage", "document", "code", "instruction", "response", "src", "source"]
            
            for col in possible_columns:
                if col in example:
                    text_column = col
                    break
            
            if text_column is None:
                # Handle the-stack dataset format
                if "blob_id" in example and "src_encoding" in example:
                    blob_id = example["blob_id"]
                    src_encoding = example.get("src_encoding", "utf-8")
                    text = self._download_stack_content(blob_id, src_encoding)
                else:
                    logger.warning(f"No text column found in sample: {list(example.keys())}")
                    return None
            else:
                text = example[text_column]
                if isinstance(text, list):
                    text = " ".join(str(item) for item in text)
                text = str(text)
            
            # Apply minimum length filter
            min_length = self.stage_processing.get('min_tokens_per_sample', 50)
            if len(text.strip()) < min_length:
                return None
            
            return text
            
        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            return None
    
    def _create_training_data_batch(self, total_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create training data using batch loading approach (original implementation).
        
        Args:
            total_samples: Total number of samples across all datasets
            
        Returns:
            List of training samples
        """
        try:
            # Load all datasets
            combined_dataset = self.load_all_datasets(total_samples=total_samples)
            
            # Convert to list of dictionaries
            training_data = []
            for i, example in enumerate(combined_dataset):
                training_data.append({
                    "text": example["text"],
                    "id": i
                })
            
            logger.info(f"Created {len(training_data)} training samples using batch approach")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to create batch training data: {e}")
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
    return loader.create_training_data(total_samples=max_samples)


if __name__ == "__main__":
    # Test the dataset loader
    config_path = "configs/config.yaml"
    tokenizer_path = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
    
    try:
        print("Testing HuggingFace dataset loader...")
        loader = HuggingFaceDatasetLoader(config_path, tokenizer_path)
        
        # Load a small sample for testing
        training_data = loader.create_training_data(total_samples=100)
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

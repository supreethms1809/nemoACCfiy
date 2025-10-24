#!/usr/bin/env python3
"""
NeMo Megatron Data Loader for ModularModel Training

This module provides NeMo Megatron-based data loading capabilities for efficient
large-scale training with proper data sharding and memory optimization.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NeMo Megatron imports
try:
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
    from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import IndexedDataset, MMapIndexedDataset
    from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
    from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
    NEMO_MEGATRON_COMPONENTS_AVAILABLE = True
except ImportError as e:
    NEMO_MEGATRON_COMPONENTS_AVAILABLE = False
    print(f"Warning: NeMo Megatron components not available: {e}")
    # Define dummy classes
    class GPTDataset:
        pass
    class IndexedDataset:
        pass
    class MMapIndexedDataset:
        pass
    class MegatronPretrainingSampler:
        pass
    class BlendableDataset:
        pass

# Tokenizer import
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: Transformers not available for tokenization")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MegatronDataLoader:
    """
    NeMo Megatron-based data loader for efficient large-scale training.
    
    This class provides optimized data loading using NeMo's Megatron components
    with proper data sharding, memory efficiency, and distributed training support.
    
    Supports both:
    1. Preprocessed Megatron datasets (.bin/.idx files)
    2. HuggingFace datasets (converted on-the-fly)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
        max_length: int = 2048,
        stage: str = "stage1",
        use_hf_datasets: bool = True,  # New: support HF datasets
        **kwargs
    ):
        """
        Initialize the Megatron data loader.
        
        Args:
            data_path: Path to the training data directory
            tokenizer_path: Path to the tokenizer
            max_length: Maximum sequence length
            stage: Training stage ("stage1", "stage2", "stage3")
        """
        if not NEMO_MEGATRON_COMPONENTS_AVAILABLE:
            raise RuntimeError("NeMo Megatron components not available. Please install NeMo with Megatron support.")
        
        if not TOKENIZER_AVAILABLE:
            raise RuntimeError("Transformers not available for tokenization.")
        
        self.data_path = Path(data_path)
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.stage = stage
        self.use_hf_datasets = use_hf_datasets
        
        # Load tokenizer with caching support
        from src.utils.tokenizer_manager import get_tokenizer_with_caching
        self.tokenizer = get_tokenizer_with_caching(
            tokenizer_path=tokenizer_path,
            custom_tokens=None,  # Use default special tokens
            force_download=False,
            cache_dir="tokenizers"
        )
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        
        logger.info(f"Initialized MegatronDataLoader for {stage}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Tokenizer: {self.tokenizer_path}")
        logger.info(f"Dataset mode: {'HuggingFace datasets' if use_hf_datasets else 'Preprocessed Megatron datasets'}")
    
    def _process_hf_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single HuggingFace dataset sample.
        
        Args:
            sample: Raw sample from HuggingFace dataset
            
        Returns:
            Processed sample or None if invalid
        """
        try:
            # Extract text (handle different dataset formats)
            text = None
            if 'text' in sample:
                text = sample['text']
            elif 'content' in sample:
                text = sample['content']
            elif 'code' in sample:
                text = sample['code']
            else:
                # Try to find any text field
                text_fields = [k for k in sample.keys() if isinstance(sample[k], str)]
                if text_fields:
                    text = sample[text_fields[0]]
                else:
                    text = str(sample)
            
            # Validate text
            if not text or len(text.strip()) == 0:
                return None
            
            # Tokenize
            tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            
            # Skip if too short
            if len(tokens) < 10:  # Minimum length threshold
                return None
            
            # Convert to tensor
            input_ids = torch.tensor(tokens, dtype=torch.long)
            
            # For next token prediction, labels are input_ids shifted by 1
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100  # Ignore last token in loss calculation
            
            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
                'text_length': len(tokens)
            }
            
        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            return None
    
    def create_indexed_dataset(self, data_file: str, use_mmap: bool = True) -> Union[IndexedDataset, MMapIndexedDataset]:
        """
        Create an indexed dataset from a data file.
        
        Args:
            data_file: Path to the data file
            use_mmap: Whether to use memory mapping for efficiency
            
        Returns:
            Indexed dataset
        """
        data_path = self.data_path / data_file
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if use_mmap:
            dataset = MMapIndexedDataset(str(data_path))
        else:
            dataset = IndexedDataset(str(data_path))
        
        logger.info(f"Created indexed dataset from {data_path}")
        logger.info(f"Dataset size: {len(dataset)} samples")
        
        return dataset
    
    def create_gpt_dataset(
        self,
        data_file: str,
        name: str,
        use_mmap: bool = True,
        **kwargs
    ) -> GPTDataset:
        """
        Create a GPT dataset for training.
        
        Args:
            data_file: Path to the data file
            name: Name of the dataset
            use_mmap: Whether to use memory mapping
            
        Returns:
            GPT dataset
        """
        # Create indexed dataset
        indexed_dataset = self.create_indexed_dataset(data_file, use_mmap)
        
        # Create GPT dataset
        gpt_dataset = GPTDataset(
            name=name,
            data_prefix=str(self.data_path / data_file),
            documents=indexed_dataset,
            indexed_dataset=indexed_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_length,
            **kwargs
        )
        
        logger.info(f"Created GPT dataset: {name}")
        logger.info(f"Dataset size: {len(gpt_dataset)} samples")
        
        return gpt_dataset
    
    def create_blendable_dataset(
        self,
        datasets_config: List[Dict[str, Any]],
        **kwargs
    ) -> BlendableDataset:
        """
        Create a blendable dataset from multiple datasets.
        
        Args:
            datasets_config: List of dataset configurations
            
        Returns:
            Blendable dataset
        """
        datasets = []
        weights = []
        
        for config in datasets_config:
            dataset = self.create_gpt_dataset(
                data_file=config["data_file"],
                name=config["name"],
                **kwargs
            )
            datasets.append(dataset)
            weights.append(config.get("weight", 1.0))
        
        blendable_dataset = BlendableDataset(
            datasets=datasets,
            weights=weights,
            **kwargs
        )
        
        logger.info(f"Created blendable dataset with {len(datasets)} datasets")
        
        return blendable_dataset
    
    def create_hf_dataset_for_megatron(
        self,
        dataset_name: str = "mlfoundations/dclm-baseline-1.0",
        max_samples: Optional[int] = None,
        **kwargs
    ) -> GPTDataset:
        """
        Create a GPT dataset from HuggingFace datasets for Megatron training.
        
        This method converts HuggingFace datasets to the format expected by NeMo Megatron
        while maintaining the efficiency benefits of Megatron training.
        
        Args:
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to use
            
        Returns:
            GPT dataset compatible with Megatron training
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("HuggingFace datasets not available. Please install datasets library.")
        
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        
        # Load HuggingFace dataset
        try:
            hf_dataset = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            logger.warning(f"Streaming failed for {dataset_name}, trying regular load: {e}")
            hf_dataset = load_dataset(dataset_name, split="train", streaming=False)
        
        # Convert to list of samples (with memory optimization)
        samples = []
        logger.info(f"Loading samples from HuggingFace dataset...")
        
        for i, sample in enumerate(hf_dataset):
            if max_samples and i >= max_samples:
                break
            
            # Process sample immediately to save memory
            processed_sample = self._process_hf_sample(sample)
            if processed_sample is not None:
                samples.append(processed_sample)
            
            # Log progress for large datasets
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} samples...")
        
        logger.info(f"Loaded {len(samples)} processed samples from HuggingFace dataset")
        
        # Create a simple dataset wrapper that works with Megatron
        class HFDatasetWrapper:
            def __init__(self, processed_samples):
                self.samples = processed_samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                # Return pre-processed sample
                return self.samples[idx]
        
        # Create wrapper dataset
        wrapper_dataset = HFDatasetWrapper(samples)
        
        # Create a simple GPT dataset that uses our wrapper
        class SimpleGPTDataset(GPTDataset):
            def __init__(self, wrapper_dataset, name="hf_dataset"):
                self.wrapper_dataset = wrapper_dataset
                self.name = name
            
            def __len__(self):
                return len(self.wrapper_dataset)
            
            def __getitem__(self, idx):
                return self.wrapper_dataset[idx]
        
        gpt_dataset = SimpleGPTDataset(wrapper_dataset, name=f"hf_{dataset_name.replace('/', '_')}")
        
        logger.info(f"Created HuggingFace-based GPT dataset with {len(gpt_dataset)} samples")
        return gpt_dataset
    
    def create_data_loader(
        self,
        dataset: Union[GPTDataset, BlendableDataset],
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create a data loader for the dataset.
        
        Args:
            dataset: The dataset to load
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            Data loader
        """
        # Create Megatron pretraining sampler for distributed training
        sampler = MegatronPretrainingSampler(
            dataset=dataset,
            consumed_samples=0,  # Start from beginning
            micro_batch_size=batch_size,
            data_parallel_rank=0,  # Will be set by distributed training
            data_parallel_size=1,  # Will be set by distributed training
        )
        
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
        logger.info(f"Created data loader with batch size {batch_size}")
        logger.info(f"Number of workers: {num_workers}")
        
        return data_loader
    
    def setup_train_val_datasets(
        self,
        train_data_file: str = "train.bin",
        val_data_file: str = "val.bin",
        batch_size: int = 8,
        num_workers: int = 8,
        hf_dataset_name: str = "mlfoundations/dclm-baseline-1.0",
        max_samples: Optional[int] = None,
        **kwargs
    ) -> tuple[DataLoader, DataLoader]:
        """
        Setup training and validation datasets.
        
        Args:
            train_data_file: Training data file (for preprocessed datasets)
            val_data_file: Validation data file (for preprocessed datasets)
            batch_size: Batch size
            num_workers: Number of worker processes
            hf_dataset_name: HuggingFace dataset name (for HF datasets)
            max_samples: Maximum number of samples to use
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Setting up train and validation datasets...")
        
        if self.use_hf_datasets:
            logger.info("Using HuggingFace datasets for Megatron training")
            
            # Create training dataset from HuggingFace
            self.train_dataset = self.create_hf_dataset_for_megatron(
                dataset_name=hf_dataset_name,
                max_samples=max_samples,
                **kwargs
            )
            
            # For validation, try to use a separate validation split if available
            try:
                # Try to load validation split
                from datasets import load_dataset
                val_dataset = load_dataset(hf_dataset_name, split="validation", streaming=True)
                logger.info("Using separate validation split from HuggingFace dataset")
                
                # Create validation dataset from validation split
                val_samples = []
                val_max_samples = max_samples // 4 if max_samples else 1000
                
                for i, sample in enumerate(val_dataset):
                    if i >= val_max_samples:
                        break
                    processed_sample = self._process_hf_sample(sample)
                    if processed_sample is not None:
                        val_samples.append(processed_sample)
                
                # Create validation wrapper
                class HFDatasetWrapper:
                    def __init__(self, processed_samples):
                        self.samples = processed_samples
                    def __len__(self):
                        return len(self.samples)
                    def __getitem__(self, idx):
                        return self.samples[idx]
                
                val_wrapper = HFDatasetWrapper(val_samples)
                
                # Create validation GPT dataset
                class SimpleGPTDataset(GPTDataset):
                    def __init__(self, wrapper_dataset, name="hf_val_dataset"):
                        self.wrapper_dataset = wrapper_dataset
                        self.name = name
                    def __len__(self):
                        return len(self.wrapper_dataset)
                    def __getitem__(self, idx):
                        return self.wrapper_dataset[idx]
                
                self.val_dataset = SimpleGPTDataset(val_wrapper, name=f"hf_val_{hf_dataset_name.replace('/', '_')}")
                logger.info(f"Created validation dataset with {len(self.val_dataset)} samples")
                
            except Exception as e:
                logger.warning(f"Could not load validation split, using training subset: {e}")
                # Fallback: use a subset of training data for validation
                self.val_dataset = self.create_hf_dataset_for_megatron(
                    dataset_name=hf_dataset_name,
                    max_samples=max_samples // 4 if max_samples else 1000,  # 25% for validation
                    **kwargs
                )
            
        else:
            logger.info("Using preprocessed Megatron datasets")
            
            # Create training dataset from preprocessed files
            self.train_dataset = self.create_gpt_dataset(
                data_file=train_data_file,
                name="train",
                **kwargs
            )
            
            # Create validation dataset from preprocessed files
            self.val_dataset = self.create_gpt_dataset(
                data_file=val_data_file,
                name="val",
                **kwargs
            )
        
        # Create data loaders
        train_loader = self.create_data_loader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        val_loader = self.create_data_loader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        logger.info("✅ Train and validation datasets setup complete")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Dataset type: {'HuggingFace' if self.use_hf_datasets else 'Preprocessed Megatron'}")
        
        return train_loader, val_loader
    
    def setup_blendable_datasets(
        self,
        datasets_config: List[Dict[str, Any]],
        batch_size: int = 8,
        num_workers: int = 8,
        **kwargs
    ) -> tuple[DataLoader, DataLoader]:
        """
        Setup blendable datasets for training.
        
        Args:
            datasets_config: List of dataset configurations
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Setting up blendable datasets...")
        
        # Create blendable dataset
        blendable_dataset = self.create_blendable_dataset(
            datasets_config=datasets_config,
            **kwargs
        )
        
        # Create data loader
        data_loader = self.create_data_loader(
            dataset=blendable_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        logger.info("✅ Blendable datasets setup complete")
        logger.info(f"Total samples: {len(blendable_dataset)}")
        
        return data_loader, None  # No separate validation for blendable datasets
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded datasets.
        
        Returns:
            Dataset information dictionary
        """
        info = {
            "data_path": str(self.data_path),
            "tokenizer_path": self.tokenizer_path,
            "max_length": self.max_length,
            "stage": self.stage,
            "train_dataset_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_dataset_size": len(self.val_dataset) if self.val_dataset else 0,
        }
        
        return info


def create_megatron_data_loader(
    data_path: str,
    tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
    max_length: int = 2048,
    stage: str = "stage1",
    **kwargs
) -> MegatronDataLoader:
    """
    Factory function to create a Megatron data loader.
    
    Args:
        data_path: Path to the training data directory
        tokenizer_path: Path to the tokenizer
        max_length: Maximum sequence length
        stage: Training stage
        
    Returns:
        Megatron data loader instance
    """
    return MegatronDataLoader(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        stage=stage,
        **kwargs
    )

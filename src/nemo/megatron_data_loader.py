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
    from nemo.collections.nlp.data.language_modeling.megatron.megatron_dataset import MegatronDataset
    NEMO_MEGATRON_AVAILABLE = True
except ImportError as e:
    NEMO_MEGATRON_AVAILABLE = False
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
    class MegatronDataset:
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
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
        max_length: int = 2048,
        stage: str = "stage1",
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
        if not NEMO_MEGATRON_AVAILABLE:
            raise RuntimeError("NeMo Megatron components not available. Please install NeMo with Megatron support.")
        
        if not TOKENIZER_AVAILABLE:
            raise RuntimeError("Transformers not available for tokenization.")
        
        self.data_path = Path(data_path)
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.stage = stage
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        
        logger.info(f"Initialized MegatronDataLoader for {stage}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Max length: {self.max_length}")
        logger.info(f"Tokenizer: {self.tokenizer_path}")
    
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
    
    def create_data_loader(
        self,
        dataset: Union[GPTDataset, BlendableDataset, MegatronDataset],
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
        **kwargs
    ) -> tuple[DataLoader, DataLoader]:
        """
        Setup training and validation datasets.
        
        Args:
            train_data_file: Training data file
            val_data_file: Validation data file
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Setting up train and validation datasets...")
        
        # Create training dataset
        self.train_dataset = self.create_gpt_dataset(
            data_file=train_data_file,
            name="train",
            **kwargs
        )
        
        # Create validation dataset
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

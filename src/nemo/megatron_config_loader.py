#!/usr/bin/env python3
"""
NeMo Megatron Configuration Loader

This module provides configuration loading and management for NeMo Megatron-based
training with optimized settings for large-scale model training.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MegatronConfigLoader:
    """
    Configuration loader for NeMo Megatron training.
    
    This class handles loading and processing configuration files specifically
    optimized for NeMo Megatron-based training with proper data loading,
    distributed training, and memory optimization settings.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the Megatron configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = None
        self.megatron_config = None
        
        # Load configuration
        self._load_config()
        
        logger.info(f"Initialized MegatronConfigLoader with config: {self.config_path}")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
    
    def create_megatron_config(
        self,
        model_config_key: str = "model_config_1.8B",
        stage: str = "stage1",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a NeMo Megatron configuration from the loaded config.
        
        Args:
            model_config_key: Model configuration key
            stage: Training stage
            **kwargs: Additional configuration overrides
            
        Returns:
            Megatron configuration dictionary
        """
        if not self.config:
            raise RuntimeError("Configuration not loaded. Call _load_config() first.")
        
        # Get model configuration
        model_configs = self.config.get("model_configs", {})
        if model_config_key not in model_configs:
            raise ValueError(f"Model configuration '{model_config_key}' not found in config")
        
        model_config = model_configs[model_config_key]
        
        # Get training configuration
        training_config = self.config.get("training", {})
        
        # Get data configuration
        data_config = self.config.get("data", {})
        
        # Get distributed configuration
        distributed_config = self.config.get("distributed", {})
        
        # Create Megatron-specific configuration
        megatron_config = {
            # Model architecture
            "vocab_size": model_config.get("vocab_size", 32000),
            "hidden_size": model_config.get("hidden_size", 768),
            "num_hidden_layers": model_config.get("num_hidden_layers", 12),
            "num_attention_heads": model_config.get("num_attention_heads", 12),
            "num_kv_heads": model_config.get("num_kv_heads", None),
            "intermediate_size": model_config.get("intermediate_size", 3072),
            "max_position_embeddings": model_config.get("max_position_embeddings", 512),
            "num_reasoning_vectors": model_config.get("num_reasoning_vectors", 8),
            "pool_type": model_config.get("pool_type", "mean"),
            "tie_weights": model_config.get("tie_weights", False),
            "freeze_embedder_decoder": model_config.get("freeze_embedder_decoder", True),
            "attention_type": model_config.get("attention_type", "gqa"),
            "mlp_type": model_config.get("mlp_type", "gated"),
            "use_flash_attention": model_config.get("use_flash_attention", True),
            "disable_flash_attention": model_config.get("disable_flash_attention", False),
            
            # Training stage
            "stage": stage,
            
            # Training configuration
            "learning_rate": float(training_config.get("learning_rate", 1e-6)),
            "weight_decay": float(training_config.get("weight_decay", 0.01)),
            "warmup_steps": int(training_config.get("warmup_steps", 1000)),
            "max_steps": int(training_config.get("max_steps", 100000)),
            "batch_size": int(training_config.get("batch_size", 8)),
            "sequence_length": int(training_config.get("sequence_length", 2048)),
            "max_grad_norm": float(training_config.get("max_grad_norm", 1.5)),
            "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 2)),
            "gradient_checkpointing": training_config.get("gradient_checkpointing", True),
            "mixed_precision": training_config.get("mixed_precision", "bf16"),
            "precision": training_config.get("mixed_precision", "bf16"),
            
            # Megatron-specific training settings
            "megatron_learning_rate": float(training_config.get("megatron_learning_rate", 1e-6)),
            "megatron_weight_decay": float(training_config.get("megatron_weight_decay", 0.01)),
            "megatron_warmup_steps": int(training_config.get("megatron_warmup_steps", 1000)),
            "megatron_max_steps": int(training_config.get("megatron_max_steps", 100000)),
            "megatron_batch_size": int(training_config.get("megatron_batch_size", 8)),
            "megatron_sequence_length": int(training_config.get("megatron_sequence_length", 2048)),
            "megatron_gradient_accumulation_steps": int(training_config.get("megatron_gradient_accumulation_steps", 2)),
            "megatron_gradient_checkpointing": training_config.get("megatron_gradient_checkpointing", True),
            "megatron_mixed_precision": training_config.get("megatron_mixed_precision", "bf16"),
            
            # Gradient clipping
            "gradient_clip_val": float(training_config.get("gradient_clip_val", 1.0)),
            "gradient_clip_algorithm": training_config.get("gradient_clip_algorithm", "norm"),
            
            # Training monitoring and control
            "log_every_n_steps": int(training_config.get("log_every_n_steps", 10)),
            "val_check_interval": int(training_config.get("val_check_interval", 1000)),
            "save_every_n_steps": int(training_config.get("save_every_n_steps", 5000)),
            "save_top_k": int(training_config.get("save_top_k", 3)),
            "monitor": training_config.get("monitor", "val_loss"),
            "mode": training_config.get("mode", "min"),
            "patience": int(training_config.get("patience", 3)),
            
            # Data configuration
            "data_path": data_config.get("data_path", "./data"),
            "tokenizer_path": data_config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom"),
            "max_samples": data_config.get("max_samples", None),
            "train_data_file": data_config.get("train_data_file", "train.bin"),
            "val_data_file": data_config.get("val_data_file", "val.bin"),
            "use_mmap": data_config.get("use_mmap", True),
            
            # Megatron data settings
            "megatron_data_path": data_config.get("megatron_data_path", "./data"),
            "megatron_tokenizer_path": data_config.get("megatron_tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom"),
            "megatron_max_length": int(data_config.get("megatron_max_length", 2048)),
            "megatron_train_data_file": data_config.get("megatron_train_data_file", "train.bin"),
            "megatron_val_data_file": data_config.get("megatron_val_data_file", "val.bin"),
            "megatron_use_mmap": data_config.get("megatron_use_mmap", True),
            
            # HuggingFace dataset settings for Megatron
            "megatron_use_hf_datasets": data_config.get("megatron_use_hf_datasets", True),
            "megatron_hf_dataset_name": data_config.get("megatron_hf_dataset_name", "mlfoundations/dclm-baseline-1.0"),
            "megatron_max_samples": data_config.get("megatron_max_samples", None),
            
            # Distributed training configuration
            "strategy": distributed_config.get("strategy", "auto"),
            "devices": distributed_config.get("devices", "auto"),
            "num_nodes": int(distributed_config.get("num_nodes", 1)),
            "tensor_model_parallel_size": int(distributed_config.get("tensor_model_parallel_size", 1)),
            "pipeline_model_parallel_size": int(distributed_config.get("pipeline_model_parallel_size", 1)),
            "data_parallel_size": int(distributed_config.get("data_parallel_size", 1)),
            "use_fsdp": distributed_config.get("use_fsdp", False),
            "use_ddp": distributed_config.get("use_ddp", True),
            "use_grad_scaler": distributed_config.get("use_grad_scaler", True),
            "use_model_parallel": distributed_config.get("use_model_parallel", False),
            
            # Output configuration
            "checkpoint_dir": f"outputs/checkpoints/megatron_{stage}",
            "model_output_dir": "outputs/models",
            "model_output_name": f"modular_model_megatron_{stage}",
            "log_dir": "outputs/logs",
            
            # Wandb configuration
            "wandb": self.config.get("wandb", {}),
            
            # Logging configuration
            "logging": self.config.get("logging", {}),
            
            # Optimizer and scheduler configurations
            "optimizer": training_config.get("optimizer", {}),
            "scheduler": training_config.get("scheduler", {}),
            
            # Megatron-specific optimizer settings
            "megatron_optimizer": training_config.get("megatron_optimizer", {}),
            "megatron_scheduler": training_config.get("megatron_scheduler", {}),
        }
        
        # Apply any overrides from kwargs
        megatron_config.update(kwargs)
        
        logger.info(f"Created Megatron configuration for {model_config_key} - {stage}")
        logger.info(f"Model size: {megatron_config['hidden_size']} hidden, {megatron_config['num_hidden_layers']} layers")
        logger.info(f"Training: {megatron_config['max_steps']} steps, batch size {megatron_config['batch_size']}")
        logger.info(f"Data: {megatron_config['data_path']}, max length {megatron_config['megatron_max_length']}")
        logger.info(f"Distributed: {megatron_config['strategy']}, {megatron_config['num_nodes']} nodes")
        
        return megatron_config
    
    def get_megatron_data_config(self, stage: str = "stage1") -> Dict[str, Any]:
        """
        Get Megatron-specific data configuration.
        
        Args:
            stage: Training stage
            
        Returns:
            Data configuration dictionary
        """
        data_config = self.config.get("data", {})
        
        megatron_data_config = {
            "data_path": data_config.get("megatron_data_path", "./data"),
            "tokenizer_path": data_config.get("megatron_tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom"),
            "max_length": int(data_config.get("megatron_max_length", 2048)),
            "train_data_file": data_config.get("megatron_train_data_file", "train.bin"),
            "val_data_file": data_config.get("megatron_val_data_file", "val.bin"),
            "use_mmap": data_config.get("megatron_use_mmap", True),
            "stage": stage,
        }
        
        return megatron_data_config
    
    def get_megatron_training_config(self, stage: str = "stage1") -> Dict[str, Any]:
        """
        Get Megatron-specific training configuration.
        
        Args:
            stage: Training stage
            
        Returns:
            Training configuration dictionary
        """
        training_config = self.config.get("training", {})
        
        megatron_training_config = {
            "learning_rate": float(training_config.get("megatron_learning_rate", 1e-6)),
            "weight_decay": float(training_config.get("megatron_weight_decay", 0.01)),
            "warmup_steps": int(training_config.get("megatron_warmup_steps", 1000)),
            "max_steps": int(training_config.get("megatron_max_steps", 100000)),
            "batch_size": int(training_config.get("megatron_batch_size", 8)),
            "sequence_length": int(training_config.get("megatron_sequence_length", 2048)),
            "gradient_accumulation_steps": int(training_config.get("megatron_gradient_accumulation_steps", 2)),
            "gradient_checkpointing": training_config.get("megatron_gradient_checkpointing", True),
            "mixed_precision": training_config.get("megatron_mixed_precision", "bf16"),
            "precision": training_config.get("megatron_mixed_precision", "bf16"),
            "stage": stage,
        }
        
        return megatron_training_config
    
    def get_megatron_distributed_config(self) -> Dict[str, Any]:
        """
        Get Megatron-specific distributed configuration.
        
        Returns:
            Distributed configuration dictionary
        """
        distributed_config = self.config.get("distributed", {})
        
        megatron_distributed_config = {
            "strategy": distributed_config.get("strategy", "auto"),
            "devices": distributed_config.get("devices", "auto"),
            "num_nodes": int(distributed_config.get("num_nodes", 1)),
            "tensor_model_parallel_size": int(distributed_config.get("tensor_model_parallel_size", 1)),
            "pipeline_model_parallel_size": int(distributed_config.get("pipeline_model_parallel_size", 1)),
            "data_parallel_size": int(distributed_config.get("data_parallel_size", 1)),
            "use_fsdp": distributed_config.get("use_fsdp", False),
            "use_ddp": distributed_config.get("use_ddp", True),
            "use_grad_scaler": distributed_config.get("use_grad_scaler", True),
            "use_model_parallel": distributed_config.get("use_model_parallel", False),
        }
        
        return megatron_distributed_config
    
    def validate_megatron_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate Megatron configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "learning_rate", "max_steps", "batch_size", "data_path", "tokenizer_path"
        ]
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate numeric fields
        numeric_fields = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "learning_rate", "max_steps", "batch_size"
        ]
        
        for field in numeric_fields:
            if not isinstance(config[field], (int, float)):
                logger.error(f"Field {field} must be numeric, got {type(config[field])}")
                return False
        
        # Validate paths
        data_path = Path(config["data_path"])
        if not data_path.exists():
            logger.warning(f"Data path does not exist: {data_path}")
        
        tokenizer_path = Path(config["tokenizer_path"])
        if not tokenizer_path.exists():
            logger.warning(f"Tokenizer path does not exist: {tokenizer_path}")
        
        logger.info("✅ Megatron configuration validation passed")
        return True


def create_megatron_config_from_existing(
    model_config_key: str = "model_config_1.8B",
    stage: str = "stage1",
    config_path: str = "configs/config.yaml",
    **kwargs
) -> Dict[str, Any]:
    """
    Factory function to create a Megatron configuration from existing config.
    
    Args:
        model_config_key: Model configuration key
        stage: Training stage
        config_path: Path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        Megatron configuration dictionary
    """
    loader = MegatronConfigLoader(config_path)
    config = loader.create_megatron_config(model_config_key, stage, **kwargs)
    
    # Validate configuration
    if not loader.validate_megatron_config(config):
        raise ValueError("Invalid Megatron configuration")
    
    return config


def main():
    """Main function for testing configuration loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Megatron Configuration Loader")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--model_config", type=str, default="model_config_1.8B", help="Model configuration key")
    parser.add_argument("--stage", type=str, default="stage1", help="Training stage")
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_megatron_config_from_existing(
            model_config_key=args.model_config,
            stage=args.stage,
            config_path=args.config
        )
        
        print("✅ Megatron configuration created successfully!")
        print(f"Model: {args.model_config} - {args.stage}")
        print(f"Hidden size: {config['hidden_size']}")
        print(f"Layers: {config['num_hidden_layers']}")
        print(f"Max steps: {config['max_steps']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Data path: {config['data_path']}")
        print(f"Strategy: {config['strategy']}")
        
    except Exception as e:
        print(f"❌ Error creating Megatron configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

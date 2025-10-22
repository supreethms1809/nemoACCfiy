"""
Configuration loader for integrating existing model_config and run_config with NeMo.
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Import tokenizer for dynamic vocab size
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: transformers not available, vocab_size will use default value")


class ConfigLoader:
    """Load and integrate configurations from model_config and run_config directories."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            base_path: Base path to the project root. If None, uses current working directory.
        """
        if base_path is None:
            # Try to find the project root from current location
            current_path = Path.cwd()
            if current_path.name == "nemo":
                base_path = current_path.parent.parent  # Go up to project root
            elif current_path.name == "src":
                base_path = current_path.parent  # Go up to project root
            else:
                base_path = current_path
        
        self.base_path = Path(base_path)
        self.configs_path = self.base_path / "configs"
        # Fallback to old structure if new structure doesn't exist
        self.model_config_path = self.base_path / "model_config"
        self.run_config_path = self.base_path / "run_config"
        
        # Cache for tokenizer and vocab size
        self._tokenizer = None
        self._vocab_size = None
    
    def get_vocab_size(self, tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom") -> int:
        """
        Get vocabulary size dynamically from tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer (local or HuggingFace model name)
            
        Returns:
            Vocabulary size
        """
        if not TOKENIZER_AVAILABLE:
            print("Warning: transformers not available, using default vocab_size: 151676")
            return 151676
        
        # Check if we have a cached vocab size
        if self._vocab_size is not None:
            return self._vocab_size
        
        try:
            # Try local tokenizer first
            local_path = self.base_path / tokenizer_path
            if local_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(local_path))
                print(f"✅ Loaded local tokenizer from: {local_path}")
            else:
                # Fallback to HuggingFace model
                self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
                print(f"✅ Loaded tokenizer from HuggingFace: Qwen/Qwen3-Coder-30B-A3B-Instruct")
            
            self._vocab_size = len(self._tokenizer)
            print(f"📊 Dynamic vocab_size: {self._vocab_size}")
            return self._vocab_size
            
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            print("Using default vocab_size: 151676")
            self._vocab_size = 151676
            return self._vocab_size
    
    def load_model_config(self, config_key: str = "model_config_1.8B") -> Dict[str, Any]:
        """
        Load model configuration from config.yaml.
        
        Args:
            config_key: Key for the model configuration to load
            
        Returns:
            Dictionary containing model configuration
        """
        # Try unified config first
        unified_config_file = self.configs_path / "config.yaml"
        
        if unified_config_file.exists():
            with open(unified_config_file, 'r') as f:
                unified_config = yaml.safe_load(f)
            
            model_configs = unified_config.get("model_configs", {})
            if config_key in model_configs:
                return model_configs[config_key]
            else:
                available_keys = list(model_configs.keys())
                raise KeyError(f"Config key '{config_key}' not found in unified config. Available keys: {available_keys}")
        
        # Fallback to old config.json structure
        config_file = self.configs_path / "config.json"
        if not config_file.exists():
            config_file = self.model_config_path / "config.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Model config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            all_configs = json.load(f)
        
        if config_key not in all_configs:
            available_keys = list(all_configs.keys())
            raise KeyError(f"Config key '{config_key}' not found. Available keys: {available_keys}")
        
        return all_configs[config_key]
    
    def load_training_config(self, stage: str = "stage1") -> Dict[str, Any]:
        """
        Load training configuration from config.yaml.
        
        Args:
            stage: Training stage to load configuration for
            
        Returns:
            Dictionary containing training configuration
        """
        # Try unified config first
        unified_config_file = self.configs_path / "config.yaml"
        
        if unified_config_file.exists():
            with open(unified_config_file, 'r') as f:
                unified_config = yaml.safe_load(f)
            
            training_stages = unified_config.get("training_stages", {})
            if stage in training_stages:
                return training_stages[stage]
            else:
                available_stages = list(training_stages.keys())
                raise KeyError(f"Stage '{stage}' not found in unified config. Available stages: {available_stages}")
        
        # Fallback to old default_training_config.yaml structure
        config_file = self.configs_path / "default_training_config.yaml"
        if not config_file.exists():
            config_file = self.run_config_path / "default_training_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Training config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        if stage not in all_configs:
            available_stages = list(all_configs.keys())
            raise KeyError(f"Stage '{stage}' not found. Available stages: {available_stages}")
        
        return all_configs[stage]
    
    def load_qwen_config(self) -> Dict[str, Any]:
        """
        Load Qwen-specific configuration from qwen_config.yaml.
        
        Returns:
            Dictionary containing Qwen configuration
        """
        config_file = self.model_config_path / "qwen_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Qwen config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def create_nemo_model_config(self, 
                                model_config_key: str = "model_config_1.8B",
                                stage: str = "stage1") -> Dict[str, Any]:
        """
        Create a NeMo-compatible model configuration by combining existing configs.
        
        Args:
            model_config_key: Key for the model configuration
            stage: Training stage
            
        Returns:
            Dictionary containing NeMo-compatible model configuration
        """
        # Load base model configuration
        model_config = self.load_model_config(model_config_key)
        
        # Load training configuration for the stage
        training_config = self.load_training_config(stage)
        
        # Extract decoder config
        decoder_config = model_config.get("decoder_config", {})
        
        # Create NeMo-compatible configuration
        nemo_config = {
            # Model architecture - use dynamic vocab size from tokenizer
            "vocab_size": self.get_vocab_size(),
            "hidden_size": decoder_config.get("hidden_size", 2048),
            "num_attention_heads": decoder_config.get("num_attention_heads", 16),
            "num_key_value_heads": decoder_config.get("num_key_value_heads", 8),
            "num_hidden_layers": decoder_config.get("num_hidden_layers", 28),
            "intermediate_size": decoder_config.get("intermediate_size", 6144),
            "max_position_embeddings": decoder_config.get("max_position_embeddings", 40960),
            "hidden_act": decoder_config.get("hidden_act", "silu"),
            "rms_norm_eps": decoder_config.get("rms_norm_eps", 1e-06),
            "initializer_range": decoder_config.get("initializer_range", 0.02),
            "use_cache": decoder_config.get("use_cache", True),
            
            # Training stage
            "training_stage": stage,
            
            # Pooling configuration
            "pool_type": model_config.get("pool_type", "attention"),
            "num_memory_vectors": 10,  # Default number of memory vectors
            
            # Dropout configuration
            "hidden_dropout_prob": model_config.get("hidden_dropout_prob", 0.0),
            "attention_probs_dropout_prob": decoder_config.get("attention_probs_dropout_prob", 0.1),
            
            # Attention configuration
            "attention_type": model_config.get("attention_type", "vanilla"),
            "attention_bias": decoder_config.get("attention_bias", False),
            
            # MLP configuration
            "mlp_type": model_config.get("mlp_type", "mlp_gated"),
            
            # Training configuration from run_config
            "training_backend": training_config.get("training", {}).get("training_backend", "lightning"),
            "learning_rate": float(training_config.get("training", {}).get("learning_rate", 1e-6)),
            "batch_size": int(training_config.get("training", {}).get("batch_size", 8)),
            "sequence_length": training_config.get("training", {}).get("sequence_length", 2048),
            "max_epochs": training_config.get("training", {}).get("epochs", 3),
            "max_grad_norm": training_config.get("training", {}).get("max_grad_norm", 1.5),
            "gradient_accumulation_steps": training_config.get("training", {}).get("gradient_accumulation_steps", 2),
            "gradient_checkpointing": training_config.get("training", {}).get("gradient_checkpointing", True),
            "mixed_precision": training_config.get("training", {}).get("mixed_precision", "bf16"),
            "precision": training_config.get("training", {}).get("mixed_precision", "bf16"),  # Alias for compatibility
            
            # Gradient clipping configuration
            "gradient_clip_val": training_config.get("training", {}).get("gradient_clip_val", 1.0),
            "gradient_clip_algorithm": training_config.get("training", {}).get("gradient_clip_algorithm", "norm"),
            
            # Training monitoring and control
            "log_every_n_steps": training_config.get("training", {}).get("log_every_n_steps", 10),
            "val_check_interval_steps": training_config.get("training", {}).get("val_check_interval_steps", 5000),
            
            # Checkpointing configuration
            "save_every_n_steps": training_config.get("training", {}).get("checkpointing", {}).get("save_every_n_steps", 1000),
            "save_top_k": training_config.get("training", {}).get("checkpointing", {}).get("save_top_k", 3),
            "monitor": training_config.get("training", {}).get("checkpointing", {}).get("monitor", "val_loss"),
            "mode": training_config.get("training", {}).get("checkpointing", {}).get("mode", "min"),
            "filename": training_config.get("training", {}).get("checkpointing", {}).get("filename", "checkpoint-{step:06d}-{val_loss:.4f}"),
            "auto_insert_metric_name": training_config.get("training", {}).get("checkpointing", {}).get("auto_insert_metric_name", False),
            
            # Tokenizer configuration
            "tokenizer_path": training_config.get("model", {}).get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom"),
            
            # Dataset configuration
            "data_path": training_config.get("data", {}).get("data_path", ""),
            "max_samples": training_config.get("data", {}).get("max_samples", None),
            
            # Megatron HuggingFace dataset settings
            "megatron_use_hf_datasets": training_config.get("data", {}).get("megatron_use_hf_datasets", True),
            "megatron_hf_dataset_name": training_config.get("data", {}).get("megatron_hf_dataset_name", "mlfoundations/dclm-baseline-1.0"),
            "megatron_max_samples": training_config.get("data", {}).get("megatron_max_samples", None),
            
            # Flash attention
            "use_flash_attention": training_config.get("model", {}).get("use_flash_attention", True),
            
            # Output configuration - use top-level outputs directory
            "checkpoint_dir": training_config.get("checkpoint_dir", "outputs/checkpoints"),
            "model_output_dir": training_config.get("model_output_dir", "outputs/models"),
            "model_output_name": training_config.get("model_output_name", f"{stage}_model.pth"),
            "log_dir": training_config.get("log_dir", "outputs/logs"),
            
            # Wandb configuration
            "wandb": training_config.get("wandb", {}),
            
            # Logging configuration
            "logging": training_config.get("logging", {}),
            
            # Optimizer and scheduler configurations
            "optimizer": training_config.get("optimizer", {}),
            "scheduler": training_config.get("scheduler", {}),
            
            # Distributed training configuration
            "distributed": training_config.get("distributed", {}),
        }
        
        return nemo_config
    
    def list_available_model_configs(self) -> list:
        """List all available model configuration keys."""
        # Try unified config first
        unified_config_file = self.configs_path / "config.yaml"
        
        if unified_config_file.exists():
            with open(unified_config_file, 'r') as f:
                unified_config = yaml.safe_load(f)
            model_configs = unified_config.get("model_configs", {})
            return list(model_configs.keys())
        
        # Fallback to old config.json structure
        config_file = self.configs_path / "config.json"
        if not config_file.exists():
            config_file = self.model_config_path / "config.json"
        
        if not config_file.exists():
            return []
        
        with open(config_file, 'r') as f:
            all_configs = json.load(f)
        
        return list(all_configs.keys())
    
    def list_available_stages(self) -> list:
        """List all available training stages."""
        # Try unified config first
        unified_config_file = self.configs_path / "config.yaml"
        
        if unified_config_file.exists():
            with open(unified_config_file, 'r') as f:
                unified_config = yaml.safe_load(f)
            training_stages = unified_config.get("training_stages", {})
            return list(training_stages.keys())
        
        # Fallback to old default_training_config.yaml structure
        config_file = self.configs_path / "default_training_config.yaml"
        if not config_file.exists():
            config_file = self.run_config_path / "default_training_config.yaml"
        
        if not config_file.exists():
            return []
        
        with open(config_file, 'r') as f:
            all_configs = yaml.safe_load(f)
        
        return list(all_configs.keys())


def create_nemo_config_from_existing(model_config_key: str = "model_config_1.8B",
                                   stage: str = "stage1",
                                   base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to create NeMo configuration from existing configs.
    
    Args:
        model_config_key: Key for the model configuration
        stage: Training stage
        base_path: Base path to the src directory
        
    Returns:
        Dictionary containing NeMo-compatible configuration
    """
    loader = ConfigLoader(base_path)
    return loader.create_nemo_model_config(model_config_key, stage)


if __name__ == "__main__":
    # Test the configuration loader
    loader = ConfigLoader()
    
    print("Available model configs:", loader.list_available_model_configs())
    print("Available stages:", loader.list_available_stages())
    
    # Test creating a NeMo config
    nemo_config = loader.create_nemo_model_config("model_config_1.8B", "stage1")
    print("\nNeMo config keys:", list(nemo_config.keys()))
    print("Vocab size:", nemo_config["vocab_size"])
    print("Hidden size:", nemo_config["hidden_size"])
    print("Training stage:", nemo_config["training_stage"])

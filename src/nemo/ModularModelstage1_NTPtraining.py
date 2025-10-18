#!/usr/bin/env python3
"""
ModularModel Stage1 NTP Training Script for NeMo

This script provides a single entry point for all training modes:
- Basic training (simple datasets)
- Production training (configuration-driven)
- Foundation training (NeMo native datasets)

Usage:
    python ModularModelstage1_NTPtraining.py --mode basic --stage stage1
    python ModularModelstage1_NTPtraining.py --mode production --model_config model_config_1.7B --stage stage1
    python ModularModelstage1_NTPtraining.py --mode foundation --data_path ./data --stage stage1
"""

import argparse
import sys
import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

try:
    from config_loader import ConfigLoader, create_nemo_config_from_existing
except ImportError:
    print("Warning: Could not import config loader: No module named 'config_loader'")
    ConfigLoader = None
    create_nemo_config_from_existing = None
from nemo_wrapper import create_modular_model_nemo, ModularModelConfig

# Optional imports for different training modes
try:
    import torch
    import torch.nn as nn
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
    from torch.utils.data import DataLoader, Dataset
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available. Training functionality disabled.")
    # Define dummy classes to prevent NameError
    class Dataset:
        pass
    class DataLoader:
        pass
    class ModelCheckpoint:
        pass
    class EarlyStopping:
        pass
    class LearningRateMonitor:
        pass
    class TensorBoardLogger:
        pass
    class WandbLogger:
        pass
    class LightningModule:
        pass
    # Create a dummy pl module
    class pl:
        class LightningModule:
            pass

# NeMo foundation training imports
try:
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
    from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import IndexedDataset, MMapIndexedDataset
    from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
    NEMO_DATASETS_AVAILABLE = True
except ImportError:
    NEMO_DATASETS_AVAILABLE = False
    print("Warning: NeMo datasets not available. Foundation training disabled.")
    # Define dummy classes
    class GPTDataset:
        pass
    class IndexedDataset:
        pass
    class MMapIndexedDataset:
        pass
    class MegatronPretrainingSampler:
        pass

# Tokenizer import
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: transformers not available. Tokenizer functionality disabled.")
    class AutoTokenizer:
        pass


def collate_fn(batch):
    """Custom collate function to properly batch the data by stacking tensors."""
    if not batch:
        return {}
    
    # If batch is a list of dictionaries, stack them
    if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
        batched = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            batched[key] = torch.stack(tensors, dim=0)
        return batched
    
    # If batch is already a dictionary, it means the default collate already processed it
    # but incorrectly (concatenated instead of stacked)
    if isinstance(batch, dict):
        # The default collate concatenated tensors, we need to reshape them
        batched = {}
        for key, tensor in batch.items():
            # Reshape from (batch_size * seq_len,) to (batch_size, seq_len)
            # We know the sequence length is 512 from our dataset
            seq_len = 512
            batch_size = tensor.size(0) // seq_len
            if batch_size > 0:
                batched[key] = tensor.view(batch_size, seq_len)
            else:
                # If reshaping fails, add a batch dimension
                batched[key] = tensor.unsqueeze(0)
        return batched
    
    return batch


class BasicDataset(Dataset):
    """Basic dataset for simple training scenarios."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        stage: str = "stage1",
        max_length: int = 512,
        pad_token_id: int = 0,
    ):
        self.data = data
        self.stage = stage
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.stage == "stage1":
            # Stage 1: Next token prediction
            input_ids = item["input_ids"]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            # Pad to max_length
            while len(input_ids) < self.max_length:
                input_ids.append(self.pad_token_id)
            
            # Create labels for next token prediction
            labels = input_ids[1:] + [self.pad_token_id]
            
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(self.max_length, dtype=torch.long)
            }
        
        elif self.stage == "stage2":
            # Stage 2: Full model training
            embed_input_ids = item.get("embed_input_ids", item["input_ids"])
            decoder_input_ids = item["input_ids"]
            
            if len(embed_input_ids) > self.max_length:
                embed_input_ids = embed_input_ids[:self.max_length]
            if len(decoder_input_ids) > self.max_length:
                decoder_input_ids = decoder_input_ids[:self.max_length]
            
            # Pad to max_length
            while len(embed_input_ids) < self.max_length:
                embed_input_ids.append(self.pad_token_id)
            while len(decoder_input_ids) < self.max_length:
                decoder_input_ids.append(self.pad_token_id)
            
            return {
                "embed_input_ids": torch.tensor(embed_input_ids, dtype=torch.long),
                "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
                "labels": torch.tensor(decoder_input_ids, dtype=torch.long),
                "embed_attention_mask": torch.ones(self.max_length, dtype=torch.long),
                "decoder_attention_mask": torch.ones(self.max_length, dtype=torch.long)
            }
        
        else:
            raise ValueError(f"Unknown stage: {self.stage}")


class NeMoFoundationDataset(Dataset):
    """NeMo foundation dataset wrapper for GPTDataset."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        use_mmap: bool = True,
        val_split: float = 0.1,
        is_training: bool = True
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mmap = use_mmap
        self.is_training = is_training
        
        if not NEMO_DATASETS_AVAILABLE:
            raise ImportError("NeMo datasets not available. Cannot use foundation training mode.")
        
        # Initialize NeMo GPTDataset
        self.gpt_dataset = GPTDataset(
            data_path=data_path,
            indexed_dataset=MMapIndexedDataset if use_mmap else IndexedDataset,
            tokenizer=tokenizer,
            max_seq_length=max_length,
            seed=42
        )
        
        # Calculate split
        total_samples = len(self.gpt_dataset)
        val_samples = int(total_samples * val_split)
        
        if is_training:
            self.start_idx = 0
            self.end_idx = total_samples - val_samples
        else:
            self.start_idx = total_samples - val_samples
            self.end_idx = total_samples
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        return self.gpt_dataset[actual_idx]


class ModularModelTrainingModule(pl.LightningModule):
    """PyTorch Lightning module for ModularModel training."""
    
    def __init__(
        self,
        model,
        stage: str = "stage1",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        optimizer_config: dict = None,
        scheduler_config: dict = None,
    ):
        super().__init__()
        self.model = model
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Store optimizer and scheduler configurations
        self.optimizer_config = optimizer_config or {
            "type": "AdamW",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
        self.scheduler_config = scheduler_config or {
            "type": "LinearLR",
            "start_factor": 0.1,
            "end_factor": 1.0,
            "warmup_steps": 1000,
            "interval": "step",
            "frequency": 1
        }
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Save hyperparameters (excluding the model to avoid pickle issues)
        self.save_hyperparameters(ignore=['model'])
    
    def training_step(self, batch, batch_idx):
        if self.stage == "stage1":
            # Stage 1: Next token prediction
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                # If outputs is the logits directly
                logits = outputs
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        elif self.stage == "stage2":
            # Stage 2: Full model training
            embed_input_ids = batch["embed_input_ids"]
            decoder_input_ids = batch["decoder_input_ids"]
            labels = batch["labels"]
            embed_attention_mask = batch["embed_attention_mask"]
            decoder_attention_mask = batch["decoder_attention_mask"]
            
            outputs = self.model(
                embed_input_ids=embed_input_ids,
                decoder_input_ids=decoder_input_ids,
                embed_attention_mask=embed_attention_mask,
                decoder_attention_mask=decoder_attention_mask
            )
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                # If outputs is the logits directly
                logits = outputs
            
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Same as training step but with model.eval()
        self.model.eval()
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
        self.model.train()
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Get learning rate from config or fallback to instance variable
        lr = self.learning_rate if self.learning_rate is not None else 1e-4
        
        # Create optimizer based on configuration
        optimizer_type = self.optimizer_config.get("type", "AdamW")
        
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=self.optimizer_config.get("weight_decay", 0.01),
                betas=tuple(self.optimizer_config.get("betas", [0.9, 0.999])),
                eps=self.optimizer_config.get("eps", 1e-8)
            )
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=self.optimizer_config.get("weight_decay", 0.01),
                betas=tuple(self.optimizer_config.get("betas", [0.9, 0.999])),
                eps=self.optimizer_config.get("eps", 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Create scheduler based on configuration
        scheduler_type = self.scheduler_config.get("type", "LinearLR")
        warmup_steps = self.scheduler_config.get("warmup_steps", self.warmup_steps)
        
        if scheduler_type == "LinearLR":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.scheduler_config.get("start_factor", 0.1),
                end_factor=self.scheduler_config.get("end_factor", 1.0),
                total_iters=warmup_steps
            )
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=warmup_steps
            )
        elif scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get("step_size", 1000),
                gamma=self.scheduler_config.get("gamma", 0.1)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_config.get("interval", "step"),
                "frequency": self.scheduler_config.get("frequency", 1)
            }
        }


def generate_sample_data(vocab_size: int = 1000, num_samples: int = 1000, max_length: int = 128) -> List[Dict[str, Any]]:
    """Generate sample training data."""
    import random
    
    data = []
    for _ in range(num_samples):
        # Generate random sequence with consistent length
        length = max_length  # Use consistent length to avoid padding issues
        input_ids = [random.randint(1, vocab_size - 1) for _ in range(length)]
        
        data.append({
            "input_ids": input_ids,
            "embed_input_ids": input_ids  # For stage2
        })
    
    return data


def load_real_dataset(data_path: str, tokenizer_path: str, max_samples: int = None, max_length: int = 2048) -> List[Dict[str, Any]]:
    """Load real dataset from files."""
    import json
    import os
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    data = []
    sample_count = 0
    
    # Support multiple file formats
    if os.path.isdir(data_path):
        # Directory of files
        for filename in os.listdir(data_path):
            if filename.endswith(('.txt', '.json', '.jsonl')):
                file_path = os.path.join(data_path, filename)
                file_data = _load_file(file_path, tokenizer, max_length)
                data.extend(file_data)
                sample_count += len(file_data)
                if max_samples and sample_count >= max_samples:
                    break
    else:
        # Single file
        data = _load_file(data_path, tokenizer, max_length)
    
    # Limit samples if specified
    if max_samples:
        data = data[:max_samples]
    
    logging.info(f"Loaded {len(data)} samples from {data_path}")
    return data


def _load_file(file_path: str, tokenizer, max_length: int) -> List[Dict[str, Any]]:
    """Load data from a single file."""
    import json
    data = []
    
    if file_path.endswith('.jsonl'):
        # JSONL format (one JSON object per line)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '')
                    if text:
                        tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length')
                        data.append({
                            "input_ids": tokens,
                            "embed_input_ids": tokens  # For stage2 compatibility
                        })
                except json.JSONDecodeError:
                    continue
                    
    elif file_path.endswith('.json'):
        # JSON format
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                for item in json_data:
                    text = item.get('text', '')
                    if text:
                        tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length')
                        data.append({
                            "input_ids": tokens,
                            "embed_input_ids": tokens  # For stage2 compatibility
                        })
            elif isinstance(json_data, dict):
                text = json_data.get('text', '')
                if text:
                    tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length')
                    data.append({
                        "input_ids": tokens,
                        "embed_input_ids": tokens  # For stage2 compatibility
                    })
                    
    elif file_path.endswith('.txt'):
        # Plain text format
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split into chunks if too long
            chunk_size = max_length * 4  # Rough estimate
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            for chunk in chunks:
                if chunk.strip():
                    tokens = tokenizer.encode(chunk, max_length=max_length, truncation=True, padding='max_length')
                    data.append({
                        "input_ids": tokens,
                        "embed_input_ids": tokens  # For stage2 compatibility
                    })
    
    return data


def load_processed_dataset(processed_data_path: str) -> List[Dict[str, Any]]:
    """Load pre-processed dataset from JSONL file."""
    import json
    
    data = []
    if os.path.isfile(processed_data_path) and processed_data_path.endswith('.jsonl'):
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Convert to the format expected by BasicDataset
                    data.append({
                        "input_ids": item["input_ids"],
                        "embed_input_ids": item["input_ids"]  # For stage2 compatibility
                    })
        logging.info(f"Loaded {len(data)} pre-processed samples from {processed_data_path}")
    else:
        raise ValueError(f"Processed dataset file not found or invalid format: {processed_data_path}")
    
    return data


def process_datasets_for_training(config_path: str = "configs/config.yaml", output_filename: str = "training_data", total_samples: int = 10000) -> str:
    """Process datasets for training using the dataset processor."""
    import sys
    from pathlib import Path
    
    # Add the dataset processing module to path
    ds_processing_path = Path(__file__).parent.parent / "ds_processing"
    sys.path.append(str(ds_processing_path))
    
    try:
        from dataset_processor import DatasetProcessor
        
        # Create processor
        processor = DatasetProcessor(config_path)
        
        # Load tokenizer
        tokenizer_path = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
        processor.load_tokenizer(tokenizer_path)
        
        # Process datasets
        logging.info(f"Processing datasets for training with {total_samples} samples...")
        processor.process_all_datasets(total_samples, output_filename)
        
        # Construct the output path
        output_path = f"data/processed/{output_filename}.jsonl"
        
        return output_path
        
    except ImportError as e:
        logging.error(f"Failed to import dataset processor: {e}")
        raise RuntimeError("Dataset processing not available")
    except Exception as e:
        logging.error(f"Failed to process datasets: {e}")
        raise


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_strategy(distributed_config: dict):
    """Create the appropriate training strategy based on configuration."""
    if not LIGHTNING_AVAILABLE:
        return None
    
    strategy_name = distributed_config.get("strategy", "auto")
    fsdp_config = distributed_config.get("fsdp", {})
    
    if strategy_name == "fsdp" or (strategy_name == "auto" and fsdp_config.get("enabled", False)):
        # Create FSDP strategy
        from lightning.pytorch.strategies import FSDPStrategy
        from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
        
        # Map string values to FSDP enums
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        
        backward_prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
            "NONE": None,
        }
        
        sharding_strategy = sharding_strategy_map.get(
            fsdp_config.get("sharding_strategy", "FULL_SHARD"),
            ShardingStrategy.FULL_SHARD
        )
        
        backward_prefetch = backward_prefetch_map.get(
            fsdp_config.get("backward_prefetch", "BACKWARD_PRE"),
            BackwardPrefetch.BACKWARD_PRE
        )
        
        strategy = FSDPStrategy(
            cpu_offload=fsdp_config.get("cpu_offload", False),
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
            limit_all_gathers=fsdp_config.get("limit_all_gathers", True),
            activation_checkpointing=fsdp_config.get("activation_checkpointing", True),
            use_orig_params=fsdp_config.get("use_orig_params", False),
        )
        
        logging.info(f"Created FSDP strategy with sharding_strategy={sharding_strategy}, "
                    f"cpu_offload={fsdp_config.get('cpu_offload', False)}")
        return strategy
    
    elif strategy_name == "ddp":
        # Create DDP strategy
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(
            sync_batchnorm=distributed_config.get("sync_batchnorm", False)
        )
        logging.info("Created DDP strategy")
        return strategy
    
    else:
        # Use auto strategy (PyTorch Lightning will choose automatically)
        logging.info("Using auto strategy (PyTorch Lightning will choose automatically)")
        return "auto"


def train_basic_mode(
    stage: str = "stage1",
    vocab_size: int = 1000,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_attention_heads: int = 4,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    devices: int = 1,
    precision: str = "16-mixed",
    max_epochs: int = 3,
    **kwargs
):
    """Basic training mode with simple datasets."""
    
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for basic training mode.")
    
    logging.info(f"🚀 Starting basic training: {stage}")
    
    # Load configuration for checkpoint and log paths
    try:
        from config_loader import create_nemo_config_from_existing
        config = create_nemo_config_from_existing("model_config_tiny", stage)
        checkpoint_dir = config.get("checkpoint_dir", f"outputs/checkpoints/{stage}")
        log_dir = config.get("log_dir", "outputs/logs")
    except ImportError:
        # Fallback if config loader not available
        checkpoint_dir = f"outputs/checkpoints/{stage}"
        log_dir = "outputs/logs"
    
    # Create model
    model = create_modular_model_nemo(
        stage=stage,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        **kwargs
    )
    
    # Generate sample data
    train_data = generate_sample_data(vocab_size, 1000)
    val_data = generate_sample_data(vocab_size, 200)
    
    # Create datasets
    train_dataset = BasicDataset(train_data, stage=stage)
    val_dataset = BasicDataset(val_data, stage=stage)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Create training module
    training_module = ModularModelTrainingModule(
        model=model,
        stage=stage,
        learning_rate=learning_rate,
        max_steps=len(train_loader) * max_epochs,
        optimizer_config=kwargs.get("optimizer_config", {}),
        scheduler_config=kwargs.get("scheduler_config", {})
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"basic_{stage}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup logger
    logger = TensorBoardLogger(log_dir, name=f"basic_{stage}")
    
    # Create strategy if distributed config is provided
    distributed_config = kwargs.get("distributed_config", {})
    strategy = create_strategy(distributed_config) if distributed_config else None
    
    # Get num_nodes from distributed config
    num_nodes = distributed_config.get("num_nodes", 1)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,  # Default 10 epochs for basic mode
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(training_module, train_loader, val_loader)
    
    logging.info("✅ Basic training completed successfully!")
    return trainer, training_module


def train_production_mode(
    model_config_key: str = "model_config_1.7B",
    stage: str = "stage1",
    devices: int = 1,
    precision: str = "bf16-mixed",
    **kwargs
):
    """Production training mode using configuration files."""
    
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for production training mode.")
    
    if create_nemo_config_from_existing is None:
        raise RuntimeError("Config loader not available for production training mode.")
    
    logging.info(f"🚀 Starting production training: {model_config_key} - {stage}")
    
    # Load configuration
    config = create_nemo_config_from_existing(model_config_key, stage)
    
    # Override parameters if provided
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
    
    # Set default values for missing keys
    if "devices" not in config:
        config["devices"] = devices or 1
    if "precision" not in config:
        config["precision"] = precision or "bf16-mixed"
    if "batch_size" not in config:
        config["batch_size"] = kwargs.get("batch_size", 4)
    if "learning_rate" not in config:
        config["learning_rate"] = kwargs.get("learning_rate", 1e-5)
    if "checkpoint_dir" not in config:
        config["checkpoint_dir"] = "outputs/checkpoints"
    
    # Create model
    model = create_modular_model_nemo(**config)
    
    # Load data - prioritize processed datasets, then real data, then sample data
    vocab_size = config["vocab_size"]
    data_path = config.get("data_path", None)
    tokenizer_path = config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom")
    use_processed_datasets = kwargs.get("use_processed_datasets", True)
    total_samples = kwargs.get("total_samples", 10000)
    
    train_data = None
    val_data = None
    
    # Try to use processed datasets first
    if use_processed_datasets:
        try:
            processed_data_path = f"data/processed/training_data.jsonl"
            if os.path.exists(processed_data_path):
                logging.info(f"Loading processed dataset from: {processed_data_path}")
                all_data = load_processed_dataset(processed_data_path)
            else:
                logging.info("Processed dataset not found. Processing datasets for training...")
                processed_data_path = process_datasets_for_training(
                    config_path="configs/config.yaml",
                    output_filename="training_data",
                    total_samples=total_samples
                )
                all_data = load_processed_dataset(processed_data_path)
            
            # Split into train/validation (80/20 split)
            split_idx = int(len(all_data) * 0.8)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples from processed datasets")
        except Exception as e:
            logging.warning(f"Failed to load processed datasets: {e}. Trying real dataset...")
            train_data = None
            val_data = None
    
    # Fallback to real dataset if processed datasets failed
    if train_data is None and data_path and os.path.exists(data_path):
        logging.info(f"Loading real dataset from: {data_path}")
        try:
            all_data = load_real_dataset(data_path, tokenizer_path, max_length=config.get("sequence_length", 2048))
            
            # Split into train/validation (80/20 split)
            split_idx = int(len(all_data) * 0.8)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples from real dataset")
        except Exception as e:
            logging.warning(f"Failed to load real dataset: {e}. Using sample data.")
            train_data = generate_sample_data(vocab_size, 1000)
            val_data = generate_sample_data(vocab_size, 200)
    
    # Final fallback to sample data
    if train_data is None:
        logging.info("Using sample data for training.")
        train_data = generate_sample_data(vocab_size, 1000)
        val_data = generate_sample_data(vocab_size, 200)
    
    # Create datasets
    train_dataset = BasicDataset(train_data, stage=stage)
    val_dataset = BasicDataset(val_data, stage=stage)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Create training module with optimizer and scheduler configs
    training_module = ModularModelTrainingModule(
        model=model,
        stage=stage,
        learning_rate=config["learning_rate"],
        max_steps=len(train_loader) * config["max_epochs"],
        optimizer_config=config.get("optimizer", {}),
        scheduler_config=config.get("scheduler", {})
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config["checkpoint_dir"],
            filename=f"production_{model_config_key}_{stage}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup logger
    log_dir = config.get("log_dir", "outputs/logs")
    logger = TensorBoardLogger(log_dir, name=f"production_{model_config_key}_{stage}")
    
    # Create strategy based on distributed configuration
    distributed_config = config.get("distributed", {})
    strategy = create_strategy(distributed_config)
    
    # Get devices and num_nodes from distributed config or fallback to config
    devices = distributed_config.get("devices", config.get("devices", "auto"))
    num_nodes = distributed_config.get("num_nodes", 1)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        devices=devices,
        num_nodes=num_nodes,
        precision=config["precision"],
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.get("max_grad_norm", 1.0),
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(training_module, train_loader, val_loader)
    
    logging.info("✅ Production training completed successfully!")
    return trainer, training_module


def train_foundation_mode(
    stage: str = "stage1",
    data_path: str = "./data",
    tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
    max_length: int = 2048,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    devices: int = 1,
    precision: str = "bf16-mixed",
    **kwargs
):
    """Foundation training mode using NeMo native datasets."""
    
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for foundation training mode.")
    
    if not NEMO_DATASETS_AVAILABLE:
        raise RuntimeError("NeMo datasets not available for foundation training mode.")
    
    if not TOKENIZER_AVAILABLE:
        raise RuntimeError("Transformers not available for foundation training mode.")
    
    logging.info(f"🚀 Starting foundation training: {stage}")
    
    # Load configuration for checkpoint and log paths
    try:
        from config_loader import create_nemo_config_from_existing
        config = create_nemo_config_from_existing("model_config_tiny", stage)
        checkpoint_dir = config.get("checkpoint_dir", f"outputs/checkpoints/{stage}")
        log_dir = config.get("log_dir", "outputs/logs")
    except ImportError:
        # Fallback if config loader not available
        checkpoint_dir = f"outputs/checkpoints/{stage}"
        log_dir = "outputs/logs"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create datasets
    train_dataset = NeMoFoundationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=True
    )
    
    val_dataset = NeMoFoundationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        is_training=False
    )
    
    # Create model (using vocab size from tokenizer)
    model = create_modular_model_nemo(
        stage=stage,
        vocab_size=len(tokenizer),
        **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Create training module
    training_module = ModularModelTrainingModule(
        model=model,
        stage=stage,
        learning_rate=learning_rate,
        max_steps=len(train_loader) * 3,  # Default 3 epochs for foundation mode
        optimizer_config=kwargs.get("optimizer_config", {}),
        scheduler_config=kwargs.get("scheduler_config", {})
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"foundation_{stage}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup logger
    logger = TensorBoardLogger(log_dir, name=f"foundation_{stage}")
    
    # Create strategy if distributed config is provided
    distributed_config = kwargs.get("distributed_config", {})
    strategy = create_strategy(distributed_config) if distributed_config else None
    
    # Get num_nodes from distributed config
    num_nodes = distributed_config.get("num_nodes", 1)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=3,  # Default 3 epochs for foundation mode
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(training_module, train_loader, val_loader)
    
    logging.info("✅ Foundation training completed successfully!")
    return trainer, training_module


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(description="Unified NeMo ModularModel Training")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="production",
                       choices=["basic", "production", "foundation"],
                       help="Training mode: basic, production, or foundation")
    
    # Common arguments
    parser.add_argument("--stage", type=str, default="stage1",
                       choices=["stage0", "stage1", "stage2"],
                       help="Training stage")
    
    
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of devices")
    
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["16-mixed", "32", "bf16-mixed"],
                       help="Training precision")
    
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    # Basic mode arguments
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="Vocabulary size (basic mode)")
    
    parser.add_argument("--hidden_size", type=int, default=256,
                       help="Hidden size (basic mode)")
    
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of layers (basic mode)")
    
    parser.add_argument("--num_attention_heads", type=int, default=4,
                       help="Number of attention heads (basic mode)")
    
    # Production mode arguments
    parser.add_argument("--model_config", type=str, default="model_config_1.7B",
                       help="Model configuration key (production mode)")
    
    # Foundation mode arguments
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Data path (foundation mode)")
    
    parser.add_argument("--tokenizer_path", type=str, 
                       default="tokenizers/qwen3-coder-30b-a3b-instruct-custom",
                       help="Tokenizer path (foundation mode)")
    
    # Dataset processing arguments
    parser.add_argument("--use_processed_datasets", action="store_true", default=True,
                       help="Use processed datasets for training (production mode)")
    
    parser.add_argument("--no_processed_datasets", action="store_true", default=False,
                       help="Disable processed datasets and use real/sample data")
    
    parser.add_argument("--total_samples", type=int, default=10000,
                       help="Total number of samples to process for training")
    
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length (foundation mode)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logging.info(f"🎯 Starting unified training in {args.mode} mode")
    
    try:
        if args.mode == "basic":
            trainer, module = train_basic_mode(
                stage=args.stage,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_attention_heads=args.num_attention_heads,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size or 4,
                devices=args.devices,
                precision=args.precision
            )
        
        elif args.mode == "production":
            # Handle processed datasets flag
            use_processed_datasets = args.use_processed_datasets and not args.no_processed_datasets
            
            trainer, module = train_production_mode(
                model_config_key=args.model_config,
                stage=args.stage,
                devices=args.devices,
                precision=args.precision,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                use_processed_datasets=use_processed_datasets,
                total_samples=args.total_samples
            )
        
        elif args.mode == "foundation":
            trainer, module = train_foundation_mode(
                stage=args.stage,
                data_path=args.data_path,
                tokenizer_path=args.tokenizer_path,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size or 4,
                devices=args.devices,
                precision=args.precision
            )
        
        logging.info("🎉 Training completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

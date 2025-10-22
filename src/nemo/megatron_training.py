#!/usr/bin/env python3
"""
NeMo Megatron Training Module for ModularModel

This module provides NeMo Megatron-based training capabilities with optimized
data loading, distributed training, and memory efficiency for large-scale models.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NeMo imports
try:
    from nemo.core.classes import NeuralModule
    from nemo.core.neural_types import NeuralType, ChannelType, MaskType, Index, LogitsType, LossType
    from nemo.core.config import Config
    from nemo.utils import logging as nemo_logging
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo core components not available")

# NeMo Megatron training imports
try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithGradScaler
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithGradScalerWithModelParallel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithModelParallel
    from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithGradScalerWithModelParallel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithModelParallel
    from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithGradScalerWithModelParallel
    from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategyWithModelParallel
    from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
    NEMO_MEGATRON_TRAINING_AVAILABLE = True
except ImportError as e:
    NEMO_MEGATRON_TRAINING_AVAILABLE = False
    print(f"Warning: NeMo Megatron training components not available: {e}")

# PyTorch Lightning imports
try:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available")

# Import our custom components
try:
    from src.nemo.megatron_data_loader import MegatronDataLoader, create_megatron_data_loader
    from src.nemo.nemo_wrapper import create_modular_model_nemo, ModularModelConfig
    MEGATRON_DATA_LOADER_AVAILABLE = True
except ImportError:
    MEGATRON_DATA_LOADER_AVAILABLE = False
    print("Warning: Megatron data loader not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MegatronTrainingModule(pl.LightningModule):
    """
    NeMo Megatron-based training module for ModularModel.
    
    This module provides optimized training with NeMo's Megatron components
    including efficient data loading, distributed training, and memory optimization.
    """
    
    def __init__(
        self,
        model,  # NeuralModule or any model
        stage: str = "stage1",
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Megatron training module.
        
        Args:
            model: The NeMo model to train
            stage: Training stage ("stage1", "stage2", "stage3")
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
        """
        super().__init__()
        
        self.model = model
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Store configurations
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
        
        # Training metrics
        self.train_loss = 0.0
        self.val_loss = 0.0
        
        logger.info(f"Initialized MegatronTrainingModule for {stage}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Max steps: {max_steps}")
    
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with NeMo Megatron optimizations.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        # Forward pass
        outputs = self.model(**batch)
        
        # Extract loss
        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("losses", 0.0))
        else:
            loss = outputs
        
        # Calculate additional metrics
        with torch.no_grad():
            # Calculate perplexity
            perplexity = torch.exp(loss)
            
            # Calculate accuracy (for next token prediction)
            if self.stage == "stage1":
                # For stage1, we need to extract logits and labels from the batch
                if "logits" in outputs and "labels" in batch:
                    shift_logits = outputs["logits"][..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    predictions = torch.argmax(shift_logits, dim=-1)
                    accuracy = (predictions == shift_labels).float().mean()
                else:
                    accuracy = torch.tensor(0.0)
            else:
                # For other stages, use the outputs directly
                if "logits" in outputs and "labels" in batch:
                    predictions = torch.argmax(outputs["logits"], dim=-1)
                    accuracy = (predictions == batch["labels"]).float().mean()
                else:
                    accuracy = torch.tensor(0.0)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        # Forward pass
        outputs = self.model(**batch)
        
        # Extract loss
        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("losses", 0.0))
        else:
            loss = outputs
        
        # Calculate validation metrics
        with torch.no_grad():
            perplexity = torch.exp(loss)
            
            # Calculate accuracy
            if self.stage == "stage1":
                if "logits" in outputs and "labels" in batch:
                    shift_logits = outputs["logits"][..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    predictions = torch.argmax(shift_logits, dim=-1)
                    accuracy = (predictions == shift_labels).float().mean()
                else:
                    accuracy = torch.tensor(0.0)
            else:
                if "logits" in outputs and "labels" in batch:
                    predictions = torch.argmax(outputs["logits"], dim=-1)
                    accuracy = (predictions == batch["labels"]).float().mean()
                else:
                    accuracy = torch.tensor(0.0)
        
        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step."""
        total_norm = 0
        param_count = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.log("grad_norm", total_norm, prog_bar=True, on_step=True)
    
    def configure_optimizers(self):
        """
        Configure optimizers and schedulers with NeMo Megatron optimizations.
        
        Returns:
            Optimizer and scheduler configuration
        """
        # Get trainable parameters
        trainable_params = self.model.get_trainable_parameters()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            **self.optimizer_config
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def create_megatron_strategy(distributed_config: Dict[str, Any]) -> str:
    """
    Create a NeMo Megatron strategy based on distributed configuration.
    
    Args:
        distributed_config: Distributed training configuration
        
    Returns:
        Strategy string or object
    """
    if not NEMO_MEGATRON_TRAINING_AVAILABLE:
        logger.warning("NeMo Megatron training not available, using default strategy")
        return "auto"
    
    strategy_type = distributed_config.get("strategy", "auto")
    devices = distributed_config.get("devices", "auto")
    num_nodes = distributed_config.get("num_nodes", 1)
    
    if strategy_type == "fsdp":
        return NLPFSDPStrategy()
    elif strategy_type == "ddp":
        return NLPDDPStrategy()
    elif strategy_type == "ddp_with_grad_scaler":
        return NLPDDPStrategyWithGradScaler()
    elif strategy_type == "ddp_with_model_parallel":
        return NLPDDPStrategyWithModelParallel()
    elif strategy_type == "ddp_with_grad_scaler_model_parallel":
        return NLPDDPStrategyWithGradScalerWithModelParallel()
    else:
        logger.info(f"Using default strategy: {strategy_type}")
        return strategy_type


def train_megatron_mode(
    model_config_key: str = "model_config_1.8B",
    stage: str = "stage1",
    data_path: str = "./data",
    tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
    max_length: int = 2048,
    learning_rate: float = 1e-6,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_steps: int = 100000,
    batch_size: int = 8,
    devices: int = 1,
    num_nodes: int = 1,
    precision: str = "bf16-mixed",
    gradient_clip_val: float = 1.0,
    gradient_clip_algorithm: str = "norm",
    log_every_n_steps: int = 10,
    val_check_interval: int = 1000,
    save_every_n_steps: int = 5000,
    save_top_k: int = 3,
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 3,
    output_dir: str = "./outputs",
    resume_from_checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    benchmark: bool = True,
    **kwargs
):
    """
    Train a ModularModel using NeMo Megatron-based training.
    
    Args:
        model_config_key: Model configuration key
        stage: Training stage
        data_path: Path to training data
        tokenizer_path: Path to tokenizer
        max_length: Maximum sequence length
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        max_steps: Maximum number of training steps
        batch_size: Batch size
        devices: Number of devices
        num_nodes: Number of nodes
        precision: Training precision
        gradient_clip_val: Gradient clipping value
        gradient_clip_algorithm: Gradient clipping algorithm
        log_every_n_steps: Logging frequency
        val_check_interval: Validation interval
        save_every_n_steps: Checkpoint saving interval
        save_top_k: Number of best checkpoints to keep
        monitor: Metric to monitor
        mode: Monitor mode
        patience: Early stopping patience
        output_dir: Output directory
        resume_from_checkpoint: Checkpoint to resume from
        seed: Random seed
        deterministic: Whether to use deterministic training
        benchmark: Whether to use cuDNN benchmark
        
    Returns:
        Tuple of (trainer, training_module)
    """
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for Megatron training.")
    
    if not MEGATRON_DATA_LOADER_AVAILABLE:
        raise RuntimeError("Megatron data loader not available.")
    
    if not NEMO_AVAILABLE:
        raise RuntimeError("NeMo not available for Megatron training.")
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # Set deterministic training
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    
    # Set mixed precision
    if precision == "bf16-mixed":
        torch.set_float32_matmul_precision('high')
    
    logger.info(f"🚀 Starting NeMo Megatron training for {stage}")
    logger.info(f"📊 Training Method: NeMo Megatron")
    logger.info(f"🔧 Framework: NeMo Megatron + Optimized Data Loading")
    logger.info(f"Model config: {model_config_key}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Precision: {precision}")
    
    # Create model
    model = create_modular_model_nemo(
        stage=stage,
        model_config_key=model_config_key,
        **kwargs
    )
    
    # Create Megatron data loader
    data_loader = create_megatron_data_loader(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        stage=stage,
        use_hf_datasets=True,  # Enable HuggingFace dataset support
        **kwargs
    )
    
    # Setup datasets (supports both HF and preprocessed datasets)
    train_loader, val_loader = data_loader.setup_train_val_datasets(
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        hf_dataset_name=kwargs.get("hf_dataset_name", "mlfoundations/dclm-baseline-1.0"),
        max_samples=kwargs.get("max_samples", None)
    )
    
    # Create training module
    training_module = MegatronTrainingModule(
        model=model,
        stage=stage,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        optimizer_config=kwargs.get("optimizer_config", {}),
        scheduler_config=kwargs.get("scheduler_config", {})
    )
    
    # Setup callbacks
    checkpoint_dir = f"{output_dir}/checkpoints/{stage}"
    log_dir = f"{output_dir}/logs"
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"megatron-{stage}-{{step:06d}}-{{val_loss:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        every_n_train_steps=save_every_n_steps,
        save_last=False,
        auto_insert_metric_name=False,
    )
    
    # Create callbacks
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Add early stopping if validation is available
    if val_loader is not None:
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Setup logger
    logger_instance = TensorBoardLogger(log_dir, name=f"megatron_{model_config_key}_{stage}")
    
    # Create distributed strategy
    distributed_config = kwargs.get("distributed_config", {})
    strategy = create_megatron_strategy(distributed_config)
    
    # Create trainer
    trainer = pl.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger_instance,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=deterministic,
        benchmark=benchmark,
    )
    
    # Train
    if resume_from_checkpoint:
        logger.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.fit(training_module, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    else:
        logger.info("🚀 Starting training from scratch...")
        trainer.fit(training_module, train_loader, val_loader)
    
    logger.info("✅ NeMo Megatron training completed successfully!")
    return trainer, training_module


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeMo Megatron Training for ModularModel")
    
    # Model configuration
    parser.add_argument("--model_config", type=str, default="model_config_1.8B", help="Model configuration key")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2", "stage3"], help="Training stage")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default="./data", help="Path to training data")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/qwen3-coder-30b-a3b-instruct-custom", help="Path to tokenizer")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Hardware configuration
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")
    
    # Training options
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm", help="Gradient clipping algorithm")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--val_check_interval", type=int, default=1000, help="Validation interval")
    parser.add_argument("--save_every_n_steps", type=int, default=5000, help="Checkpoint saving interval")
    parser.add_argument("--save_top_k", type=int, default=3, help="Number of best checkpoints to keep")
    parser.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor")
    parser.add_argument("--mode", type=str, default="min", help="Monitor mode")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume from")
    
    # Other options
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic training")
    parser.add_argument("--benchmark", action="store_true", help="Use cuDNN benchmark")
    
    args = parser.parse_args()
    
    # Train
    trainer, training_module = train_megatron_mode(**vars(args))
    
    print("✅ NeMo Megatron training completed successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ModularModel Stage1 NTP Training Script for NeMo

This script provides a single entry point for all training modes:
- Basic training (simple datasets)
- Production training (configuration-driven)
- Foundation training (NeMo native datasets)

Usage with conda:
    conda activate nemo
    # Lightning backend with HuggingFace datasets
    python train.py --mode production --model_config model_config_tiny --stage stage1 --no_processed_datasets
    # Megatron backend (automatically uses HuggingFace datasets)
    python train.py --mode production --training_backend megatron --model_config model_config_tiny --stage stage1

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

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src structure
try:
    from src.nemo.config_loader import ConfigLoader, create_nemo_config_from_existing
except ImportError:
    # Silent fallback - config loader is optional
    ConfigLoader = None
    create_nemo_config_from_existing = None

try:
    from src.nemo.huggingface_dataset_loader import HuggingFaceDatasetLoader, HuggingFaceDatasetWrapper
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: HuggingFace dataset loader not available")
    HF_DATASETS_AVAILABLE = False
    HuggingFaceDatasetLoader = None
    HuggingFaceDatasetWrapper = None

try:
    from src.nemo.nemo_wrapper import create_modular_model_nemo, ModularModelConfig
    NEMO_AVAILABLE = True
except ImportError:
    print("Warning: NeMo wrapper not available")
    NEMO_AVAILABLE = False
    create_modular_model_nemo = None
    ModularModelConfig = None

# Optional imports for different training modes
try:
    import torch
    import torch.nn as nn
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar, DeviceStatsMonitor
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
    class RichProgressBar:
        def __init__(self, **kwargs):
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

# NeMo Megatron training imports
try:
    from src.nemo.megatron_training import train_megatron_mode
    from src.nemo.megatron_config_loader import create_megatron_config_from_existing
    MEGATRON_TRAINING_AVAILABLE = True
except ImportError:
    MEGATRON_TRAINING_AVAILABLE = False
    print("Warning: NeMo Megatron training not available.")
    train_megatron_mode = None
    create_megatron_config_from_existing = None

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
        optimizer_config: dict = None,
        scheduler_config: dict = None,
    ):
        super().__init__()
        self.model = model
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
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
        
        # Calculate additional metrics
        with torch.no_grad():
            # Calculate perplexity
            perplexity = torch.exp(loss)
            
            # Calculate accuracy (for next token prediction)
            if self.stage == "stage1":
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                predictions = torch.argmax(shift_logits, dim=-1)
                accuracy = (predictions == shift_labels).float().mean()
            else:
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
        
        # Log metrics with progress bar display
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log learning rate (this should make it appear in progress bar)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Same as training step but with model.eval()
        self.model.eval()
        with torch.no_grad():
            if self.stage == "stage1":
                # Stage 1: Next token prediction
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate validation metrics
                perplexity = torch.exp(loss)
                predictions = torch.argmax(shift_logits, dim=-1)
                accuracy = (predictions == shift_labels).float().mean()
            
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
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Calculate validation metrics
                perplexity = torch.exp(loss)
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
            
            else:
                raise ValueError(f"Unknown stage: {self.stage}")
        
        self.model.train()
        
        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step"""
        # Calculate gradient norm
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
        # Get learning rate from config or fallback to instance variable
        lr = self.learning_rate if self.learning_rate is not None else 1e-4
        
        # Create optimizer based on configuration
        optimizer_type = self.optimizer_config.get("type", "AdamW")
        
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=float(lr),
                weight_decay=float(self.optimizer_config.get("weight_decay", 0.01)),
                betas=tuple(self.optimizer_config.get("betas", [0.9, 0.999])),
                eps=float(self.optimizer_config.get("eps", 1e-8))
            )
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=float(lr),
                weight_decay=float(self.optimizer_config.get("weight_decay", 0.01)),
                betas=tuple(self.optimizer_config.get("betas", [0.9, 0.999])),
                eps=float(self.optimizer_config.get("eps", 1e-8))
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


def process_datasets_for_training(config_path: str = "configs/config.yaml", output_filename: str = "training_data", total_samples: int = 10000, base_path: Optional[str] = None) -> str:
    """Process datasets for training using the dataset processor or HuggingFace datasets."""
    import sys
    from pathlib import Path
    import yaml
    
    # Resolve config path relative to base_path if provided
    if base_path is not None:
        config_path = str(Path(base_path) / config_path)
    
    # Check if we should use HuggingFace datasets
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if data_path is None (use HuggingFace datasets)
        training_stages = config.get("training_stages", {})
        stage1_config = training_stages.get("stage1", {})
        data_config = stage1_config.get("data", {})
        data_path = data_config.get("data_path")
        
        if data_path is None and HF_DATASETS_AVAILABLE:
            logging.info("Using HuggingFace datasets for training...")
            return _process_huggingface_datasets(config_path, output_filename, total_samples, base_path)
        
    except Exception as e:
        logging.warning(f"Could not check config for HuggingFace datasets: {e}")
    
    # Fallback to original dataset processor
    logging.info("Using local dataset processor...")
    
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


def _process_huggingface_datasets(config_path: str, output_filename: str, total_samples: int, base_path: Optional[str] = None) -> str:
    """Process HuggingFace datasets for training."""
    from pathlib import Path
    import json
    
    # Resolve paths
    if base_path is not None:
        config_path = str(Path(base_path) / config_path)
        output_dir = Path(base_path) / "data" / "processed"
    else:
        output_dir = Path("data") / "processed"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load HuggingFace datasets
    tokenizer_path = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
    if base_path is not None:
        tokenizer_path = str(Path(base_path) / tokenizer_path)
    
    loader = HuggingFaceDatasetLoader(config_path, tokenizer_path, stage="stage1")
    training_data = loader.create_training_data(max_samples=total_samples)
    
    # Save to JSONL file
    output_path = output_dir / f"{output_filename}.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logging.info(f"Saved {len(training_data)} HuggingFace dataset samples to {output_path}")
    return str(output_path)


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
    resume_from_checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    benchmark: bool = True,
    **kwargs
):
    """Basic training mode with simple datasets."""
    
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for basic training mode.")
    
    # Set environment for reproducibility
    if seed is not None:
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    
    # Optimize for Tensor Cores (RTX 6000 Ada Generation)
    torch.set_float32_matmul_precision('high')
    
    logging.info(f"🚀 Starting basic training: {stage}")
    logging.info(f"📊 Training Method: PyTorch Lightning")
    logging.info(f"🔧 Framework: PyTorch Lightning + Sample Data")
    logging.info(f"📈 Progress Bar: PyTorch Lightning Default")
    
    # Load configuration for checkpoint and log paths
    if create_nemo_config_from_existing is not None:
        try:
            config = create_nemo_config_from_existing("model_config_tiny", stage)
            checkpoint_dir = config.get("checkpoint_dir", f"outputs/checkpoints/{stage}")
            log_dir = config.get("log_dir", "outputs/logs")
        except Exception:
            # Fallback if config loading fails
            checkpoint_dir = f"outputs/checkpoints/{stage}"
            log_dir = "outputs/logs"
    else:
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
        collate_fn=collate_fn
    )
    
    # Create training module
    training_module = ModularModelTrainingModule(
        model=model,
        stage=stage,
        learning_rate=learning_rate,
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
    strategy = create_strategy(distributed_config) if distributed_config else "auto"
    
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
    if resume_from_checkpoint:
        logging.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
    trainer.fit(training_module, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    
    logging.info("✅ Basic training completed successfully!")
    return trainer, training_module


def train_production_mode(
    model_config_key: str = "model_config_1.7B",
    stage: str = "stage1",
    devices: int = 1,
    precision: str = "bf16-mixed",
    base_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    benchmark: bool = True,
    **kwargs
):
    """Production training mode using configuration files."""
    
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for production training mode.")
    
    if create_nemo_config_from_existing is None:
        raise RuntimeError("Config loader not available for production training mode.")
    
    # Set environment for reproducibility
    if seed is not None:
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    
    # Optimize for Tensor Cores (RTX 6000 Ada Generation)
    torch.set_float32_matmul_precision('high')
    
    logging.info(f"🚀 Starting production training: {model_config_key} - {stage}")
    logging.info(f"📊 Training Method: PyTorch Lightning")
    logging.info(f"🔧 Framework: PyTorch Lightning + HuggingFace Datasets")
    logging.info(f"📈 Progress Bar: PyTorch Lightning Default (RichProgressBar disabled due to compatibility issues)")
    
    # Load configuration
    config = create_nemo_config_from_existing(model_config_key, stage, base_path)
    
    # Override parameters if provided (but don't override with None values)
    for key, value in kwargs.items():
        if key in config and value is not None:
            config[key] = value
    
    # Set default values for missing keys
    if "devices" not in config:
        config["devices"] = devices or 1
    if "precision" not in config:
        config["precision"] = precision or "bf16-mixed"
    if "batch_size" not in config:
        batch_size_arg = kwargs.get("batch_size", 4)
        config["batch_size"] = batch_size_arg if batch_size_arg is not None else 4
    if "learning_rate" not in config:
        lr_arg = kwargs.get("learning_rate", 1e-5)
        config["learning_rate"] = lr_arg if lr_arg is not None else 1e-5
    if "checkpoint_dir" not in config:
        config["checkpoint_dir"] = "outputs/checkpoints"
    
    # Create model
    model = create_modular_model_nemo(**config)
    
    # Load data - prioritize processed datasets, then real data, then sample data
    vocab_size = config["vocab_size"]
    data_path = config.get("data_path", None)
    tokenizer_path = config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom")
    # Read use_processed_datasets from config, fall back to kwargs, then default
    use_processed_datasets = kwargs.get("use_processed_datasets") if kwargs.get("use_processed_datasets") is not None else config.get("use_processed_datasets", False)
    # Read max_samples from config, fall back to kwargs, then default
    total_samples = kwargs.get("total_samples") if kwargs.get("total_samples") is not None else config.get("max_samples", 10000)
    
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
                    total_samples=total_samples,
                    base_path=base_path
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
    
    # Try HuggingFace datasets if processed datasets are disabled and no real dataset path
    if train_data is None and not use_processed_datasets and (not data_path or not os.path.exists(data_path)):
        if HF_DATASETS_AVAILABLE:
            logging.info("Using HuggingFace datasets for training...")
            try:
                # Use HuggingFace dataset loader with stage parameter
                hf_loader = HuggingFaceDatasetLoader("configs/config.yaml", stage="stage1")
                combined_dataset = hf_loader.load_all_datasets(max_samples_per_dataset=total_samples)
                
                # Convert to the format expected by BasicDataset
                all_data = []
                logging.info(f"Converting {len(combined_dataset)} samples from HuggingFace datasets...")
                
                for i in range(len(combined_dataset)):
                    sample = combined_dataset[i]
                    # Debug: log the first sample structure
                    if i == 0:
                        logging.info(f"Sample structure: {list(sample.keys())}")
                        if 'input_ids' in sample:
                            logging.info(f"Sample has input_ids with length: {len(sample['input_ids'])}")
                        else:
                            logging.info(f"Sample keys: {list(sample.keys())}")
                    
                    # The HuggingFace dataset loader already processes the data and returns input_ids
                    if 'input_ids' in sample:
                        all_data.append({
                            "input_ids": sample["input_ids"],
                            "embed_input_ids": sample["input_ids"]  # For stage2 compatibility
                        })
                    else:
                        # If no input_ids, try to find text and tokenize it
                        text = sample.get('text', '')
                        if text:
                            # Tokenize the text using the tokenizer
                            tokenizer = hf_loader.tokenizer
                            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=2048)
                            all_data.append({
                                "input_ids": tokens,
                                "embed_input_ids": tokens  # For stage2 compatibility
                            })
                
                logging.info(f"Converted {len(all_data)} samples to training format")
                
                if all_data:
                    # Split into train/validation (80/20 split)
                    split_idx = int(len(all_data) * 0.8)
                    train_data = all_data[:split_idx]
                    val_data = all_data[split_idx:]
                    
                    logging.info(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples from HuggingFace datasets")
                else:
                    logging.warning("No data loaded from HuggingFace datasets")
                    
            except Exception as e:
                logging.warning(f"Failed to load HuggingFace datasets: {e}")
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
    
    # Ensure batch_size is valid
    batch_size = config.get("batch_size", 4)
    if batch_size is None or batch_size == 0:
        batch_size = 4
        logging.warning(f"Invalid batch_size ({config.get('batch_size')}), using default value of {batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True, collate_fn=collate_fn)
    
    # Create training module with optimizer and scheduler configs
    training_module = ModularModelTrainingModule(
        model=model,
        stage=stage,
        learning_rate=config["learning_rate"],
        optimizer_config=config.get("optimizer", {}),
        scheduler_config=config.get("scheduler", {})
    )
    
    # Get training configuration (from flattened config structure)
    # The config loader flattens the training config to top level
    training_config = config  # All training params are now at top level
    
    # Setup callbacks with step-based configuration
    checkpointing_config = {
        "save_every_n_steps": config.get("save_every_n_steps", 1000),
        "save_top_k": config.get("save_top_k", 3),
        "monitor": config.get("monitor", "val_loss"),
        "mode": config.get("mode", "min"),
        "filename": config.get("filename", "checkpoint-{step:06d}-{val_loss:.4f}"),
        "auto_insert_metric_name": config.get("auto_insert_metric_name", False),
    }
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    max_epochs = config.get("max_epochs", 2)
    
    # Use step-based settings directly (more frequent monitoring)
    save_every_n_steps = config.get("save_every_n_steps", 1000)
    val_check_interval_steps = config.get("val_check_interval_steps", 5000)
    
    # Calculate total steps and adjust intervals if needed
    total_steps = max_epochs * steps_per_epoch
    
    # Adjust validation interval if it's greater than total steps
    if val_check_interval_steps > total_steps:
        val_check_interval_steps = max(1, total_steps // 4)  # Validate 4 times during training
        logging.warning(f"Adjusted val_check_interval from {config.get('val_check_interval_steps', 5000)} to {val_check_interval_steps} (total steps: {total_steps})")
    
    # Adjust save interval if it's greater than total steps
    if save_every_n_steps > total_steps:
        save_every_n_steps = max(1, total_steps // 2)  # Save 2 times during training
        logging.warning(f"Adjusted save_every_n_steps from {config.get('save_every_n_steps', 1000)} to {save_every_n_steps} (total steps: {total_steps})")
    
    # Debug: Log the actual config values being read
    logging.info(f"🔍 Config Debug:")
    logging.info(f"   - checkpointing_config: {checkpointing_config}")
    logging.info(f"   - save_every_n_steps from config: {config.get('save_every_n_steps', 'NOT_FOUND')}")
    logging.info(f"   - val_check_interval_steps from config: {config.get('val_check_interval_steps', 'NOT_FOUND')}")
    logging.info(f"   - gradient_clip_val from config: {config.get('gradient_clip_val', 'NOT_FOUND')}")
    logging.info(f"   - gradient_clip_algorithm from config: {config.get('gradient_clip_algorithm', 'NOT_FOUND')}")
    
    logging.info(f"📊 Training Configuration:")
    logging.info(f"   - Steps per epoch: {steps_per_epoch}")
    logging.info(f"   - Max epochs: {max_epochs}")
    logging.info(f"   - Total steps: {max_epochs * steps_per_epoch}")
    logging.info(f"   - Save every {save_every_n_steps} steps")
    logging.info(f"   - Validate every {val_check_interval_steps} steps")
    logging.info(f"   - Gradient clipping: {config.get('gradient_clip_val', 1.0)} ({config.get('gradient_clip_algorithm', 'norm')})")
    logging.info(f"   - Checkpoint directory: {config['checkpoint_dir']}")
    logging.info(f"   - Best checkpoints: checkpoint-{{step:06d}}-{{val_loss:.4f}}.ckpt (top {checkpointing_config.get('save_top_k', 3)})")
    logging.info(f"   - Latest checkpoints: last-checkpoint-{{step:06d}}.ckpt")
    
    # Use step-based checkpointing (direct step intervals)
    step_checkpoint = ModelCheckpoint(
        dirpath=config["checkpoint_dir"],
        filename=checkpointing_config.get("filename", "checkpoint-{step:06d}-{val_loss:.4f}"),
        monitor=checkpointing_config.get("monitor", "val_loss"),
        mode=checkpointing_config.get("mode", "min"),
        save_top_k=checkpointing_config.get("save_top_k", 3),
        every_n_train_steps=save_every_n_steps,
        auto_insert_metric_name=checkpointing_config.get("auto_insert_metric_name", False),
        save_last=False,  # Don't save last.ckpt to avoid overwriting
        save_on_train_epoch_end=False,
    )
    
    # Create a separate callback for saving the last checkpoint
    last_checkpoint = ModelCheckpoint(
        dirpath=config["checkpoint_dir"],
        filename="last-checkpoint-{step:06d}",
        every_n_train_steps=save_every_n_steps,
        save_top_k=1,  # Only keep the latest
        save_last=False,
        save_on_train_epoch_end=False,
    )
    
    # Create callbacks (skip RichProgressBar due to known issues)
    callbacks = [
        step_checkpoint,  # Best checkpoints based on val_loss
        last_checkpoint,  # Latest checkpoint with step number
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        # RichProgressBar(),  # Disabled due to "pop from empty list" error
    ]
    
    # Note: RichProgressBar disabled due to "pop from empty list" error
    # PyTorch Lightning will use its default progress bar
    
    # Setup logger
    log_dir = config.get("log_dir", "outputs/logs")
    logger = TensorBoardLogger(log_dir, name=f"production_{model_config_key}_{stage}")
    
    # Create strategy based on distributed configuration
    distributed_config = config.get("distributed", {})
    strategy = create_strategy(distributed_config)
    
    # Get devices and num_nodes from distributed config or fallback to config
    devices = distributed_config.get("devices", config.get("devices", "auto"))
    num_nodes = distributed_config.get("num_nodes", 1)
    
    # Get training duration (epoch-based) - training_config already defined above
    max_epochs = training_config.get("epochs", config.get("max_epochs", 2))
    
    # Create trainer with step-based configuration
    trainer_kwargs = {
        "devices": devices,
        "num_nodes": num_nodes,
        "precision": config["precision"],
        "strategy": strategy,
        "callbacks": callbacks,
        "logger": logger,
        "gradient_clip_val": config.get("gradient_clip_val", 1.0),
        "gradient_clip_algorithm": config.get("gradient_clip_algorithm", "norm"),
        "max_epochs": max_epochs,  # Always use epochs as primary configuration
        "log_every_n_steps": training_config.get("log_every_n_steps", 10),
        "val_check_interval": val_check_interval_steps,  # Step-based validation interval
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    logging.info(f"🚀 Training for {max_epochs} epochs ({max_epochs * steps_per_epoch} total steps) with step-based monitoring")
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Auto-detect latest checkpoint for resume
    checkpoint_path = resume_from_checkpoint
    if not checkpoint_path:
        # Try to find the latest checkpoint automatically
        checkpoint_dir = Path(config["checkpoint_dir"])
        if checkpoint_dir.exists():
            # Look for the last checkpoint file
            last_checkpoint = checkpoint_dir / "last.ckpt"
            if last_checkpoint.exists():
                checkpoint_path = str(last_checkpoint)
                logging.info(f"🔄 Auto-detected latest checkpoint: {checkpoint_path}")
            else:
                # Look for any checkpoint files and get the latest
                checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
                if checkpoint_files:
                    # Sort by modification time and get the latest
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    checkpoint_path = str(latest_checkpoint)
                    logging.info(f"🔄 Auto-detected latest checkpoint: {checkpoint_path}")
    
    # Train with resume capability
    if checkpoint_path:
        logging.info(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        try:
            trainer.fit(training_module, train_loader, val_loader, ckpt_path=checkpoint_path)
        except Exception as e:
            logging.warning(f"Failed to resume from checkpoint {checkpoint_path}: {e}")
            logging.info("Starting training from scratch...")
            trainer.fit(training_module, train_loader, val_loader)
    else:
        logging.info("🚀 Starting training from scratch...")
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
    resume_from_checkpoint: Optional[str] = None,
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
    logging.info(f"📊 Training Method: PyTorch Lightning")
    logging.info(f"🔧 Framework: PyTorch Lightning + NeMo Native Datasets")
    
    # Load configuration for checkpoint and log paths
    if create_nemo_config_from_existing is not None:
        try:
            config = create_nemo_config_from_existing("model_config_tiny", stage)
            checkpoint_dir = config.get("checkpoint_dir", f"outputs/checkpoints/{stage}")
            log_dir = config.get("log_dir", "outputs/logs")
        except Exception:
            # Fallback if config loading fails
            checkpoint_dir = f"outputs/checkpoints/{stage}"
            log_dir = "outputs/logs"
    else:
        # Fallback if config loader not available
        checkpoint_dir = f"outputs/checkpoints/{stage}"
        log_dir = "outputs/logs"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create datasets
    train_dataset = None  # TODO: Implement NeMo native dataset creation
    val_dataset = None    # TODO: Implement NeMo native dataset creation
    
    # TODO: Complete foundation training implementation
    logging.warning("Foundation training mode is not yet fully implemented")
    logging.info("Consider using 'megatron' mode for NeMo Megatron-based training")
    
    return None, None


def train_megatron_mode_wrapper(
    model_config_key: str = "model_config_1.7B",
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
    """NeMo Megatron training mode wrapper."""
    
    if not MEGATRON_TRAINING_AVAILABLE:
        raise RuntimeError("NeMo Megatron training not available. Please check NeMo installation.")
    
    if train_megatron_mode is None:
        raise RuntimeError("NeMo Megatron training function not available.")
    
    logging.info(f"🚀 Starting NeMo Megatron training: {stage}")
    logging.info(f"📊 Training Method: NeMo Megatron")
    logging.info(f"🔧 Framework: NeMo Megatron + Optimized Data Loading")
    logging.info(f"Model config: {model_config_key}")
    logging.info(f"Data path: {data_path}")
    logging.info(f"Max steps: {max_steps}")
    
    # Use the Megatron training function
    return train_megatron_mode(
        model_config_key=model_config_key,
        stage=stage,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        batch_size=batch_size,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        save_every_n_steps=save_every_n_steps,
        save_top_k=save_top_k,
        monitor=monitor,
        mode=mode,
        patience=patience,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        seed=seed,
        deterministic=deterministic,
        benchmark=benchmark,
        **kwargs
    )


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(description="Unified NeMo ModularModel Training")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="production",
                       choices=["basic", "production", "foundation"],
                       help="Training mode: basic, production, or foundation")
    
    # Training backend (for production mode)
    parser.add_argument("--training_backend", type=str, default=None,
                       choices=["lightning", "megatron"],
                       help="Training backend for production mode: lightning (PyTorch Lightning) or megatron (NeMo Megatron). If not specified, reads from config.")
    
    # Common arguments
    parser.add_argument("--stage", type=str, default="stage1",
                       choices=["stage0", "stage1", "stage2"],
                       help="Training stage")
    
    
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of devices")
    
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["16-mixed", "32", "bf16-mixed"],
                       help="Training precision")
    
    # Resume training arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from (auto-detect if not specified)")
    
    parser.add_argument("--no_resume", action="store_true",
                       help="Disable auto-resume from latest checkpoint")
    
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
    
    # Dataset processing arguments (only for Lightning backend)
    parser.add_argument("--use_processed_datasets", action="store_true", default=True,
                       help="Use processed datasets for training (Lightning backend only)")
    
    parser.add_argument("--no_processed_datasets", action="store_true", default=False,
                       help="Disable processed datasets and use HuggingFace datasets (Lightning backend only)")
    
    parser.add_argument("--total_samples", type=int, default=None,
                       help="Total number of samples to process for training (overrides config)")
    
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length (foundation mode)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Log training mode selection
    logging.info("="*60)
    logging.info(f"🎯 SELECTED TRAINING MODE: {args.mode.upper()}")
    logging.info("="*60)
    
    if args.mode == "basic":
        logging.info("📊 Training Method: PyTorch Lightning")
        logging.info("🔧 Framework: PyTorch Lightning + Sample Data")
        logging.info("💡 Use case: Quick testing and development")
    elif args.mode == "production":
        # Get training backend for logging (same logic as in production mode)
        training_backend = args.training_backend
        if training_backend is None:
            if create_nemo_config_from_existing is not None:
                try:
                    config = create_nemo_config_from_existing(args.model_config, args.stage)
                    training_backend = config.get("training_backend", "lightning")
                except Exception:
                    training_backend = "lightning"
            else:
                training_backend = "lightning"
        
        if training_backend == "lightning":
            logging.info("📊 Training Method: PyTorch Lightning")
            logging.info("🔧 Framework: PyTorch Lightning + HuggingFace Datasets")
            logging.info("💡 Use case: Production training with real datasets")
        elif training_backend == "megatron":
            logging.info("📊 Training Method: NeMo Megatron")
            logging.info("🔧 Framework: NeMo Megatron + HuggingFace Datasets")
            logging.info("💡 Use case: Large-scale, distributed training with optimizations")
    elif args.mode == "foundation":
        logging.info("📊 Training Method: PyTorch Lightning")
        logging.info("🔧 Framework: PyTorch Lightning + NeMo Native Datasets")
        logging.info("💡 Use case: NeMo native dataset training (incomplete)")
    
    logging.info("="*60)
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
            # Handle resume checkpoint
            resume_from_checkpoint = None
            if not args.no_resume:
                resume_from_checkpoint = args.resume_from_checkpoint
            
            # Get training backend from config if not provided via command line
            training_backend = args.training_backend
            if training_backend is None:
                # Load config to get training_backend
                if create_nemo_config_from_existing is not None:
                    try:
                        config = create_nemo_config_from_existing(args.model_config, args.stage)
                        training_backend = config.get("training_backend", "lightning")
                        logging.info(f"📊 Training backend from config: {training_backend}")
                    except Exception as e:
                        logging.warning(f"Could not load config for training_backend, using default: {e}")
                        training_backend = "lightning"
                else:
                    training_backend = "lightning"
            
            logging.info(f"🎯 Using training backend: {training_backend}")
            
            if training_backend == "lightning":
                # Handle processed datasets flag (only relevant for Lightning backend)
                # Command line arguments override config values
                if args.no_processed_datasets:
                    use_processed_datasets = False
                elif args.use_processed_datasets:
                    use_processed_datasets = True
                else:
                    # Use config value, default to False (use HuggingFace datasets directly)
                    use_processed_datasets = config.get("use_processed_datasets", False)
                
                # Use PyTorch Lightning backend
                trainer, module = train_production_mode(
                    model_config_key=args.model_config,
                    stage=args.stage,
                    devices=args.devices,
                    precision=args.precision,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    resume_from_checkpoint=resume_from_checkpoint,
                    use_processed_datasets=use_processed_datasets,
                    total_samples=args.total_samples
                )
            elif training_backend == "megatron":
                # Use NeMo Megatron backend
                trainer, module = train_megatron_mode_wrapper(
                    model_config_key=args.model_config,
                    stage=args.stage,
                    data_path=args.data_path,
                    tokenizer_path=args.tokenizer_path,
                    max_length=args.max_length,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size or 8,
                    devices=args.devices,
                    precision=args.precision,
                    resume_from_checkpoint=resume_from_checkpoint,
                    hf_dataset_name=args.data_path,  # Use data_path as dataset name for HF datasets
                    max_samples=args.total_samples
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
        logging.info("="*60)
        logging.info(f"✅ COMPLETED: {args.mode.upper()} MODE TRAINING")
        if args.mode == "basic":
            logging.info("📊 Training Method Used: PyTorch Lightning")
        elif args.mode == "production":
            # Get training backend for completion logging
            training_backend = args.training_backend
            if training_backend is None:
                if create_nemo_config_from_existing is not None:
                    try:
                        config = create_nemo_config_from_existing(args.model_config, args.stage)
                        training_backend = config.get("training_backend", "lightning")
                    except Exception:
                        training_backend = "lightning"
                else:
                    training_backend = "lightning"
            
            if training_backend == "lightning":
                logging.info("📊 Training Method Used: PyTorch Lightning")
            elif training_backend == "megatron":
                logging.info("📊 Training Method Used: NeMo Megatron")
        elif args.mode == "foundation":
            logging.info("📊 Training Method Used: PyTorch Lightning")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

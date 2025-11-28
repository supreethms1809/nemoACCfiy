#!/usr/bin/env python3
"""
ModularModel Stage2 Full Model Training Script for NeMo

This script implements Stage 2 training with:
- Full modular model (embedder + decoder with cross-attention)
- Decoder loaded from Stage 1 checkpoint
- New embedder initialization
- Dual loss functions: contrastive loss for embedder+attention pooling + cross_entropy loss
- Instruction tuning format with question-answer pairs
- OpenCodeReasoning and OpenMathReasoning datasets

Usage:
    python ModularModelStage2_FullModelTraining.py --mode production --model_config model_config_1.8B --stage stage2
    python ModularModelStage2_FullModelTraining.py --mode basic --stage stage2
"""

import argparse
import sys
import os
import logging
import math
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from src structure
try:
    from src.nemo.config_loader import ConfigLoader, create_nemo_config_from_existing
except ImportError:
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

# PyTorch and Lightning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar, DeviceStatsMonitor
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
    from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
    from torch.utils.data import DataLoader, Dataset
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available. Training functionality disabled.")

# HuggingFace datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available")

# Tokenizer import
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: transformers not available. Tokenizer functionality disabled.")


class ContrastiveLoss(nn.Module):
    """Contrastive loss for embedder + attention pooling training."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for reasoning embeddings.
        
        Args:
            embeddings: (batch_size, num_reasoning_vectors, hidden_size) - reasoning vectors
            labels: (batch_size,) - sample labels for contrastive learning
        """
        batch_size, num_vectors, hidden_size = embeddings.shape
        
        # Flatten reasoning vectors for contrastive learning
        # Each sample has multiple reasoning vectors, we'll use the mean
        pooled_embeddings = embeddings.mean(dim=1)  # (batch_size, hidden_size)
        
        # Normalize embeddings
        pooled_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(pooled_embeddings, pooled_embeddings.T) / self.temperature
        
        # Create labels for contrastive learning
        # For now, we'll use a simple approach where each sample is its own positive
        # In a more sophisticated setup, you might want to group similar samples
        labels = torch.arange(batch_size, device=embeddings.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class InstructionTuningDataset(Dataset):
    """Dataset for instruction tuning with question-answer pairs."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 1024,
        embed_max_length: int = 4096,
        stage: str = "stage2",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embed_max_length = embed_max_length
        self.stage = stage
        
        # Special tokens for instruction tuning
        self.instruction_token = "<|instruction|>"
        self.reasoning_token = "<|reasoning|>"
        self.response_token = "<|response|>"
        self.end_token = "<|end|>"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.stage == "stage2":
            # Stage 2: Full model training with instruction tuning format
            question = item.get("question", item.get("input", ""))
            reasoning = item.get("reasoning", item.get("analysis", ""))
            answer = item.get("answer", item.get("output", ""))
            
            # Handle empty or None reasoning
            if not reasoning or reasoning.strip() == "":
                reasoning = "No reasoning provided."
            
            # Ensure all fields are strings
            question = str(question) if question else ""
            reasoning = str(reasoning) if reasoning else "No reasoning provided."
            answer = str(answer) if answer else ""
            
            # Create instruction tuning format
            # Format: <|instruction|> {question} <|reasoning|> {reasoning} <|response|> {answer} <|end|>
            instruction_text = f"{self.instruction_token} {question} {self.reasoning_token} {reasoning} {self.response_token} {answer} {self.end_token}"
            
            # Tokenize the full instruction
            instruction_tokens = self.tokenizer.encode(
                instruction_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).squeeze(0)
            
            # Create reasoning part for embedder (question + reasoning)
            reasoning_text = f"{self.instruction_token} {question} {self.reasoning_token} {reasoning}"
            reasoning_tokens = self.tokenizer.encode(
                reasoning_text,
                max_length=self.embed_max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).squeeze(0)
            
            # Create labels (shifted input_ids for next token prediction)
            labels = instruction_tokens.clone()
            labels[:-1] = instruction_tokens[1:]  # Shift right
            labels[-1] = -100  # Ignore last token
            
            # Create attention masks
            instruction_attention_mask = (instruction_tokens != self.tokenizer.pad_token_id).long()
            reasoning_attention_mask = (reasoning_tokens != self.tokenizer.pad_token_id).long()
            
            return {
                "input_ids": instruction_tokens,
                "embed_input_ids": reasoning_tokens,
                "labels": labels,
                "attention_mask": instruction_attention_mask,
                "embed_attention_mask": reasoning_attention_mask,
                "question": question,
                "reasoning": reasoning,
                "answer": answer
            }
        
        else:
            raise ValueError(f"Unknown stage: {self.stage}")


class Stage2TrainingModule(pl.LightningModule):
    """PyTorch Lightning module for Stage 2 ModularModel training."""
    
    def __init__(
        self,
        model,
        stage: str = "stage2",
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        contrastive_weight: float = 0.1,
        cross_entropy_weight: float = 1.0,
        optimizer_config: dict = None,
        scheduler_config: dict = None,
    ):
        super().__init__()
        self.model = model
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.contrastive_weight = contrastive_weight
        self.cross_entropy_weight = cross_entropy_weight
        
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
        
        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Save hyperparameters (excluding the model to avoid pickle issues)
        self.save_hyperparameters(ignore=['model'])
    
    def training_step(self, batch, batch_idx):
        if self.stage == "stage2":
            # Stage 2: Full model training with dual losses
            input_ids = batch["input_ids"]
            embed_input_ids = batch["embed_input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            embed_attention_mask = batch["embed_attention_mask"]
            
            # Forward pass through the full modular model
            outputs = self.model(
                input_ids=input_ids,
                embed_input_ids=embed_input_ids,
                attention_mask=attention_mask,
                embed_attention_mask=embed_attention_mask
            )
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Cross-entropy loss for next token prediction
            ce_loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Contrastive loss for embedder + attention pooling
            # Get reasoning vectors from embedder
            with torch.no_grad():
                reasoning_vectors = self.model.embedder(embed_input_ids, embed_attention_mask=embed_attention_mask)
            
            # Create pseudo-labels for contrastive learning (each sample is its own positive)
            batch_size = reasoning_vectors.size(0)
            contrastive_labels = torch.arange(batch_size, device=reasoning_vectors.device)
            
            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(reasoning_vectors, contrastive_labels)
            
            # Combined loss
            total_loss = self.cross_entropy_weight * ce_loss + self.contrastive_weight * contrastive_loss
            
            # Calculate additional metrics
            with torch.no_grad():
                # Calculate perplexity
                perplexity = torch.exp(ce_loss)
                
                # Calculate accuracy (for next token prediction)
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
                
                # Calculate contrastive accuracy (simplified)
                contrastive_accuracy = torch.tensor(0.0, device=reasoning_vectors.device)  # Placeholder
        
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        
        # Log metrics with progress bar display
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_ce_loss", ce_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_contrastive_loss", contrastive_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_contrastive_accuracy", contrastive_accuracy, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log learning rate
        try:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log("lr", current_lr, prog_bar=True, on_step=True)
        except RuntimeError:
            current_lr = self.learning_rate
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.log("lr", current_lr, prog_bar=True, on_step=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Same as training step but with model.eval()
        self.model.eval()
        with torch.no_grad():
            if self.stage == "stage2":
                # Stage 2: Full model training with dual losses
                input_ids = batch["input_ids"]
                embed_input_ids = batch["embed_input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                embed_attention_mask = batch["embed_attention_mask"]
                
                # Forward pass through the full modular model
                outputs = self.model(
                    input_ids=input_ids,
                    embed_input_ids=embed_input_ids,
                    attention_mask=attention_mask,
                    embed_attention_mask=embed_attention_mask
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Cross-entropy loss for next token prediction
                ce_loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Contrastive loss for embedder + attention pooling
                reasoning_vectors = self.model.embedder(embed_input_ids, embed_attention_mask=embed_attention_mask)
                batch_size = reasoning_vectors.size(0)
                contrastive_labels = torch.arange(batch_size, device=reasoning_vectors.device)
                contrastive_loss = self.contrastive_loss(reasoning_vectors, contrastive_labels)
                
                # Combined loss
                total_loss = self.cross_entropy_weight * ce_loss + self.contrastive_weight * contrastive_loss
                
                # Calculate validation metrics
                perplexity = torch.exp(ce_loss)
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
                contrastive_accuracy = torch.tensor(0.0, device=reasoning_vectors.device)  # Placeholder
            
            else:
                raise ValueError(f"Unknown stage: {self.stage}")
        
        self.model.train()
        
        # Log validation metrics
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ce_loss", ce_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_contrastive_loss", contrastive_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_contrastive_accuracy", contrastive_accuracy, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
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
        lr = self.learning_rate if self.learning_rate is not None else 5e-6
        
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
        warmup_steps = int(self.scheduler_config.get("warmup_steps", self.warmup_steps))
        
        if scheduler_type == "LinearLR":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(self.scheduler_config.get("start_factor", 0.1)),
                end_factor=float(self.scheduler_config.get("end_factor", 1.0)),
                total_iters=warmup_steps
            )
        elif scheduler_type == "CosineAnnealingLR":
            # For cosine annealing, we need to handle warmup separately
            if warmup_steps > 0:
                # Create warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=float(self.scheduler_config.get("start_factor", 0.1)),
                    end_factor=float(self.scheduler_config.get("end_factor", 1.0)),
                    total_iters=warmup_steps
                )
                
                # Get T_max from config (represents total training steps)
                total_training_steps = int(self.scheduler_config.get("T_max", 100000))
                
                # Validate that T_max > warmup_steps
                if total_training_steps <= warmup_steps:
                    raise ValueError(
                        f"T_max ({total_training_steps}) must be greater than warmup_steps ({warmup_steps}). "
                        f"T_max represents total training steps, so it should include warmup steps."
                    )
                
                # Cosine annealing runs for (T_max - warmup_steps) steps after warmup completes
                # SequentialLR will switch to cosine scheduler at warmup_steps, and cosine scheduler
                # will run for cosine_steps steps (counting from 0 internally)
                cosine_steps = total_training_steps - warmup_steps
                
                # Create cosine annealing scheduler
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_steps,  # Number of steps for cosine annealing (after warmup)
                    eta_min=float(self.scheduler_config.get("eta_min", 1e-7))
                )
                
                # Combine warmup and cosine annealing
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                # No warmup, just cosine annealing
                # T_max represents total training steps (no warmup to subtract)
                total_training_steps = int(self.scheduler_config.get("T_max", 100000))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_training_steps,
                    eta_min=float(self.scheduler_config.get("eta_min", 1e-7))
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


def load_instruction_datasets(
    dataset_names: List[str] = ["nvidia/OpenCodeReasoning", "nvidia/OpenMathReasoning"],
    max_samples: int = 10000,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """Load instruction tuning datasets from HuggingFace."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library not available")
    
    all_data = []
    
    for dataset_name in dataset_names:
        try:
            logging.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split=split)
            
            # Convert to our format
            for i, item in enumerate(dataset):
                if i >= max_samples // len(dataset_names):
                    break
                
                # Handle different dataset formats
                if "problem" in item and "expected_answer" in item:
                    # nvidia/OpenMathReasoning format
                    sample = {
                        "question": item["problem"],
                        "reasoning": item.get("generated_solution", ""),
                        "answer": item["expected_answer"]
                    }
                elif "question" in item and "expected_answer" in item:
                    # nvidia/OpenMathInstruct format
                    sample = {
                        "question": item["question"],
                        "reasoning": item.get("generated_solution", item.get("reasoning", item.get("analysis", ""))),
                        "answer": item["expected_answer"]
                    }
                elif "question" in item and "answer" in item:
                    # Generic question-answer format (OpenCodeReasoning or other)
                    sample = {
                        "question": item["question"],
                        "reasoning": item.get("reasoning", item.get("analysis", item.get("solution", ""))),
                        "answer": item["answer"]
                    }
                elif "input" in item and "output" in item:
                    # Generic instruction format
                    sample = {
                        "question": item["input"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["output"]
                    }
                elif "query" in item and "answer" in item:
                    # Query-answer format (m-a-p/CodeFeedback-Filtered-Instruction)
                    sample = {
                        "question": item["query"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["answer"]
                    }
                elif "query" in item and "response" in item:
                    # Query-response format (meta-math/MetaMathQA)
                    sample = {
                        "question": item["query"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["response"]
                    }
                elif "problem statement" in item and "solution" in item:
                    # Problem statement-solution format (hpcgroup/hpc-instruct)
                    sample = {
                        "question": item["problem statement"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["solution"]
                    }
                else:
                    # Log unknown format for debugging
                    logging.debug(f"Unknown format in {dataset_name}: {list(item.keys())}")
                    continue
                
                all_data.append(sample)
            
            logging.info(f"Loaded {len(all_data)} samples from {dataset_name}")
            
        except Exception as e:
            logging.warning(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    logging.info(f"Total samples loaded: {len(all_data)}")
    return all_data


def load_instruction_datasets_with_percentages(
    pretraining_datasets: Dict[str, Dict[str, Any]],
    max_samples: int = 10000,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """Load instruction tuning datasets with percentage-based sampling from config."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library not available")
    
    all_data = []
    total_percentage = sum(dataset_config.get("percentage", 0) for dataset_config in pretraining_datasets.values())
    
    if total_percentage == 0:
        logging.warning("No valid dataset percentages found in config. Using equal distribution.")
        total_percentage = 100.0
        equal_percentage = 100.0 / len(pretraining_datasets)
        for dataset_name in pretraining_datasets:
            pretraining_datasets[dataset_name]["percentage"] = equal_percentage
    
    for dataset_name, dataset_config in pretraining_datasets.items():
        try:
            percentage = dataset_config.get("percentage", 0)
            subset = dataset_config.get("subset", None)
            
            # Calculate samples for this dataset based on percentage
            dataset_samples = int(max_samples * (percentage / 100.0))
            
            if dataset_samples == 0:
                logging.info(f"Skipping {dataset_name} (0% allocation)")
                continue
                
            logging.info(f"Loading dataset: {dataset_name} ({percentage}% = {dataset_samples} samples)")
            
            # Load dataset with optional subset
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Check dataset size and adjust if needed
            try:
                # Try to get dataset length (works for non-streaming datasets)
                dataset_size = len(dataset)
                if dataset_size < dataset_samples:
                    logging.warning(
                        f"  Dataset {dataset_name} has only {dataset_size:,} samples, "
                        f"less than requested {dataset_samples:,} ({percentage}%). "
                        f"Using all available samples."
                    )
                    dataset_samples = dataset_size
                elif dataset_size == dataset_samples:
                    logging.info(f"  Dataset {dataset_name} has exactly {dataset_size:,} samples")
                else:
                    logging.info(f"  Dataset {dataset_name} has {dataset_size:,} samples, taking {dataset_samples:,}")
            except (TypeError, AttributeError):
                # Streaming dataset or unknown size - proceed with requested limit
                logging.info(f"  Dataset size unknown (streaming?), will take up to {dataset_samples:,} samples")
            
            # Convert to our format
            dataset_data = []
            for i, item in enumerate(dataset):
                if i >= dataset_samples:
                    break
                
                # Handle different dataset formats
                if "problem" in item and "expected_answer" in item:
                    # nvidia/OpenMathReasoning format
                    sample = {
                        "question": item["problem"],
                        "reasoning": item.get("generated_solution", ""),
                        "answer": item["expected_answer"],
                        "dataset": dataset_name  # Track source dataset
                    }
                elif "question" in item and "expected_answer" in item:
                    # nvidia/OpenMathInstruct format
                    sample = {
                        "question": item["question"],
                        "reasoning": item.get("generated_solution", item.get("reasoning", item.get("analysis", ""))),
                        "answer": item["expected_answer"],
                        "dataset": dataset_name  # Track source dataset
                    }
                elif "question" in item and "answer" in item:
                    # Generic question-answer format (OpenCodeReasoning or other)
                    sample = {
                        "question": item["question"],
                        "reasoning": item.get("reasoning", item.get("analysis", item.get("solution", ""))),
                        "answer": item["answer"],
                        "dataset": dataset_name  # Track source dataset
                    }
                elif "input" in item and "output" in item:
                    # Generic instruction format
                    sample = {
                        "question": item["input"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["output"],
                        "dataset": dataset_name  # Track source dataset
                    }
                elif "query" in item and "answer" in item:
                    # Query-answer format (m-a-p/CodeFeedback-Filtered-Instruction)
                    sample = {
                        "question": item["query"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["answer"],
                        "dataset": dataset_name  # Track source dataset
                    }
                elif "query" in item and "response" in item:
                    # Query-response format (meta-math/MetaMathQA)
                    sample = {
                        "question": item["query"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["response"],
                        "dataset": dataset_name  # Track source dataset
                    }
                elif "problem statement" in item and "solution" in item:
                    # Problem statement-solution format (hpcgroup/hpc-instruct)
                    sample = {
                        "question": item["problem statement"],
                        "reasoning": item.get("reasoning", item.get("analysis", "")),
                        "answer": item["solution"],
                        "dataset": dataset_name  # Track source dataset
                    }
                else:
                    # Log unknown format for debugging
                    logging.debug(f"Unknown format in {dataset_name}: {list(item.keys())}")
                    continue
                
                dataset_data.append(sample)
            
            all_data.extend(dataset_data)
            requested_samples = int(max_samples * (percentage / 100.0))
            if len(dataset_data) < requested_samples:
                logging.info(
                    f"Loaded {len(dataset_data):,} samples from {dataset_name} "
                    f"(requested {requested_samples:,}, {len(dataset_data)/requested_samples*100:.1f}% of requested)"
                )
            else:
                logging.info(f"Loaded {len(dataset_data):,} samples from {dataset_name}")
            
        except Exception as e:
            logging.warning(f"Failed to load dataset {dataset_name}: {e}")
            continue

    logging.info(f"Total samples loaded: {len(all_data)}")
    
    # Validate loaded data
    validate_instruction_data(all_data)
    
    return all_data


def validate_instruction_data(data: List[Dict[str, Any]]) -> None:
    """Validate that instruction tuning data has required fields."""
    if not data:
        logging.warning("No data loaded for validation")
        return
    
    required_fields = ["question", "reasoning", "answer"]
    validation_stats = {
        "total_samples": len(data),
        "missing_fields": {field: 0 for field in required_fields},
        "empty_fields": {field: 0 for field in required_fields},
        "valid_samples": 0
    }
    
    for i, sample in enumerate(data):
        is_valid = True
        
        for field in required_fields:
            if field not in sample:
                validation_stats["missing_fields"][field] += 1
                is_valid = False
                logging.warning(f"Sample {i}: Missing field '{field}'")
            elif not sample[field] or str(sample[field]).strip() == "":
                validation_stats["empty_fields"][field] += 1
                is_valid = False
                logging.warning(f"Sample {i}: Empty field '{field}'")
        
        if is_valid:
            validation_stats["valid_samples"] += 1
    
    # Log validation results
    logging.info("📊 Data Validation Results:")
    logging.info(f"  Total samples: {validation_stats['total_samples']}")
    logging.info(f"  Valid samples: {validation_stats['valid_samples']}")
    logging.info(f"  Missing fields: {validation_stats['missing_fields']}")
    logging.info(f"  Empty fields: {validation_stats['empty_fields']}")
    
    if validation_stats["valid_samples"] == 0:
        raise RuntimeError("No valid samples found in loaded data!")
    elif validation_stats["valid_samples"] < validation_stats["total_samples"] * 0.8:
        logging.warning(f"Only {validation_stats['valid_samples']}/{validation_stats['total_samples']} samples are valid!")


def collate_fn_stage2(batch, tokenizer=None, max_length=1024, embed_max_length=4096):
    """Custom collate function for Stage 2 training."""
    if not batch:
        return {}
    
    # Extract data from batch
    input_ids = torch.stack([item["input_ids"] for item in batch])
    embed_input_ids = torch.stack([item["embed_input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    embed_attention_mask = torch.stack([item["embed_attention_mask"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "embed_input_ids": embed_input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "embed_attention_mask": embed_attention_mask,
        "questions": [item["question"] for item in batch],
        "reasonings": [item["reasoning"] for item in batch],
        "answers": [item["answer"] for item in batch]
    }


def train_basic_mode(
    stage: str = "stage2",
    vocab_size: int = 1000,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_attention_heads: int = 4,
    learning_rate: float = 5e-6,
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
    
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    logging.info(f"🚀 Starting basic Stage 2 training")
    logging.info(f"📊 Training Method: PyTorch Lightning")
    logging.info(f"🔧 Framework: PyTorch Lightning + Instruction Tuning")
    
    # Load tokenizer
    if TOKENIZER_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise RuntimeError("Tokenizer not available for basic training mode.")
    
    # Create model
    model = create_modular_model_nemo(
        stage=stage,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        **kwargs
    )
    
    # Generate sample instruction data
    train_data = generate_sample_instruction_data(1000)
    val_data = generate_sample_instruction_data(200)
    
    # Create datasets
    train_dataset = InstructionTuningDataset(train_data, tokenizer, stage=stage)
    val_dataset = InstructionTuningDataset(val_data, tokenizer, stage=stage)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=lambda batch: collate_fn_stage2(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=lambda batch: collate_fn_stage2(batch, tokenizer)
    )
    
    # Create training module
    training_module = Stage2TrainingModule(
        model=model,
        stage=stage,
        learning_rate=learning_rate,
        optimizer_config=kwargs.get("optimizer_config", {}),
        scheduler_config=kwargs.get("scheduler_config", {})
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"outputs/checkpoints/{stage}",
            filename=f"basic_{stage}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup logger
    logger = TensorBoardLogger("outputs/logs", name=f"basic_{stage}")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # Train
    if resume_from_checkpoint:
        logging.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
    trainer.fit(training_module, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    
    logging.info("✅ Basic Stage 2 training completed successfully!")
    return trainer, training_module


def train_production_mode(
    model_config_key: str = "model_config_1.8B",
    stage: str = "stage2",
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
    
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    logging.info(f"🚀 Starting production Stage 2 training: {model_config_key}")
    logging.info(f"📊 Training Method: PyTorch Lightning")
    logging.info(f"🔧 Framework: PyTorch Lightning + Instruction Tuning")
    
    # Load configuration
    config = create_nemo_config_from_existing(model_config_key, stage, base_path)
    
    # Override parameters if provided
    for key, value in kwargs.items():
        if key in config and value is not None:
            if config[key] != value:
                logging.info(f"🔄 Overriding config {key}: {config[key]} → {value}")
            config[key] = value
    
    # Set default values for missing keys
    if "devices" not in config:
        config["devices"] = devices if devices is not None else 1
    if "precision" not in config:
        config["precision"] = precision if precision is not None else "bf16-mixed"
    if "batch_size" not in config:
        batch_size_arg = kwargs.get("batch_size")
        config["batch_size"] = batch_size_arg if batch_size_arg is not None else 4
    if "learning_rate" not in config:
        lr_arg = kwargs.get("learning_rate")
        config["learning_rate"] = lr_arg if lr_arg is not None else 5e-6
    if "checkpoint_dir" not in config:
        config["checkpoint_dir"] = "outputs/checkpoints"
    
    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    if TOKENIZER_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise RuntimeError("Tokenizer not available for production training mode.")
    
    # Create model - ensure stage parameter is passed for correct model selection
    config["training_stage"] = stage
    
    # Determine if this is resume training or fresh training
    is_resume_training = resume_from_checkpoint is not None
    
    # Handle Stage 2 checkpoint loading logic
    if is_resume_training:
        # Resume Stage 2 training: Load both embedder and decoder from Stage 2 checkpoint
        logging.info(f"🔄 Resuming Stage 2 training from checkpoint: {resume_from_checkpoint}")
        logging.info("📋 Stage 2 Resume Mode: Will load both embedder and decoder from Stage 2 checkpoint")
        
        # For resume, don't load Stage 1 checkpoint - let the model load from Stage 2 checkpoint
        config["embedder_checkpoint_path"] = None
        config["freeze_embedder_decoder"] = False  # Unfreeze for resume training
        
    else:
        # Fresh Stage 2 training: Load Stage 1 checkpoint for embedder decoder, initialize new pooler
        logging.info("🆕 Fresh Stage 2 training: Loading embedder decoder from Stage 1 checkpoint")
        
        stage1_checkpoint_path = config.get("stage1_checkpoint_path")
        if stage1_checkpoint_path is None:
            # Auto-detect Stage 1 checkpoint
            stage1_checkpoint_dir = "/home/sureshm/nemoACCfiy/outputs/checkpoints/stage1"
            if os.path.exists(stage1_checkpoint_dir):
                # Look for the latest Stage 1 checkpoint
                checkpoint_files = [f for f in os.listdir(stage1_checkpoint_dir) if f.endswith('.ckpt')]
                if checkpoint_files:
                    # Sort by modification time and get the latest
                    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(stage1_checkpoint_dir, x)))
                    stage1_checkpoint_path = os.path.join(stage1_checkpoint_dir, latest_checkpoint)
                    logging.info(f"🔄 Auto-detected Stage 1 checkpoint: {stage1_checkpoint_path}")
                else:
                    logging.warning("⚠️ No Stage 1 checkpoints found. Starting with randomly initialized model.")
            else:
                logging.warning("⚠️ Stage 1 checkpoint directory not found. Starting with randomly initialized model.")
        
        # Add Stage 1 checkpoint path to config for fresh training
        if stage1_checkpoint_path:
            config["embedder_checkpoint_path"] = stage1_checkpoint_path
            config["freeze_embedder_decoder"] = config.get("freeze_embedder_decoder", True)
            logging.info(f"🔒 Will load embedder decoder from Stage 1 checkpoint: {stage1_checkpoint_path}")
            logging.info(f"🔒 Embedder decoder will be frozen: {config.get('freeze_embedder_decoder', True)}")
            logging.info("🆕 Embedder pooler will be initialized new and trainable")
        else:
            logging.warning("⚠️ No Stage 1 checkpoint found. Starting with randomly initialized embedder and decoder.")
            config["embedder_checkpoint_path"] = None
            config["freeze_embedder_decoder"] = False
    
    model = create_modular_model_nemo(**config)
    
    # Load instruction tuning datasets from config
    data_config = config.get("data", {})
    pretraining_datasets = data_config.get("pretraining_datasets", {})
    
    # Extract dataset names from config (fallback to default if not found)
    dataset_names = list(pretraining_datasets.keys()) if pretraining_datasets else ["nvidia/OpenCodeReasoning", "nvidia/OpenMathReasoning"]
    max_samples = data_config.get("processing", {}).get("total_samples", 100000)
    
    logging.info(f"Loading instruction tuning datasets from config: {dataset_names}")
    logging.info(f"Max samples per dataset: {max_samples // len(dataset_names)}")
    
    # Load datasets with proper sampling based on config percentages
    train_data = load_instruction_datasets_with_percentages(
        pretraining_datasets, 
        max_samples // 2, 
        "train"
    )
    val_data = load_instruction_datasets_with_percentages(
        pretraining_datasets, 
        max_samples // 10, 
        "validation"
    )
    
    # Create datasets
    train_dataset = InstructionTuningDataset(
        train_data, 
        tokenizer, 
        max_length=config.get("sequence_length", 1024),
        embed_max_length=config.get("embed_sequence_length", 4096),
        stage=stage
    )
    val_dataset = InstructionTuningDataset(
        val_data, 
        tokenizer, 
        max_length=config.get("sequence_length", 1024),
        embed_max_length=config.get("embed_sequence_length", 4096),
        stage=stage
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 8),
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True),
        prefetch_factor=config.get("prefetch_factor", 2),
        drop_last=True,
        collate_fn=lambda batch: collate_fn_stage2(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 8),
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True),
        prefetch_factor=config.get("prefetch_factor", 2),
        drop_last=True,
        collate_fn=lambda batch: collate_fn_stage2(batch, tokenizer)
    )
    
    # Create training module with dual loss weights from config
    training_module = Stage2TrainingModule(
        model=model,
        stage=stage,
        learning_rate=config["learning_rate"],
        contrastive_weight=config.get("contrastive_weight", 0.1),
        cross_entropy_weight=config.get("cross_entropy_weight", 1.0),
        optimizer_config=config.get("optimizer", {}),
        scheduler_config=config.get("scheduler", {})
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config["checkpoint_dir"],
            filename=f"stage2_checkpoint_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup logger
    log_dir = config.get("log_dir", "outputs/logs")
    logger = TensorBoardLogger(log_dir, name=f"production_{model_config_key}_{stage}")
    
    # Create trainer with Stage 2 specific configuration
    trainer = pl.Trainer(
        max_epochs=config.get("epochs", 3),
        devices=config.get("devices", 1),
        precision=config["precision"],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.get("gradient_clip_val", 0.8),
        gradient_clip_algorithm=config.get("gradient_clip_algorithm", "value"),
        log_every_n_steps=config.get("log_every_n_steps", 100),
        val_check_interval=config.get("val_check_interval_steps", 1000),
        accumulate_grad_batches=config.get("gradient_accumulation_steps", 2),
        gradient_checkpointing=config.get("gradient_checkpointing", True)
    )
    
    # Handle resume checkpoint auto-detection if not provided
    if resume_from_checkpoint is None:
        # Auto-detect latest Stage 2 checkpoint for resume
        stage2_checkpoint_dir = config["checkpoint_dir"]
        if os.path.exists(stage2_checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(stage2_checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoint_files:
                # Sort by modification time and get the latest
                latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(stage2_checkpoint_dir, x)))
                resume_from_checkpoint = os.path.join(stage2_checkpoint_dir, latest_checkpoint)
                logging.info(f"🔄 Auto-detected latest Stage 2 checkpoint: {resume_from_checkpoint}")
                logging.info("📋 Auto-resume Mode: Will load both embedder and decoder from Stage 2 checkpoint")
                is_resume_training = True
            else:
                logging.info("ℹ️ No Stage 2 checkpoints found. Starting fresh Stage 2 training.")
                is_resume_training = False
        else:
            logging.info("ℹ️ Stage 2 checkpoint directory not found. Starting fresh Stage 2 training.")
            is_resume_training = False
    
    # Train with appropriate checkpoint handling
    if is_resume_training and resume_from_checkpoint:
        logging.info(f"🔄 Resuming Stage 2 training from checkpoint: {resume_from_checkpoint}")
        logging.info("📋 Resume Mode: Loading both embedder and decoder from Stage 2 checkpoint")
        
        # Check if model was created for fresh training but we're resuming
        if config.get("embedder_checkpoint_path") is not None:
            logging.warning("⚠️ WARNING: Model was created for fresh training but resuming from Stage 2 checkpoint.")
            logging.warning("⚠️ This may cause issues. Consider recreating the model for resume training.")
    else:
        logging.info("🆕 Fresh Stage 2 training: Starting with Stage 1 embedder decoder + new pooler")
    
    trainer.fit(training_module, train_loader, val_loader, ckpt_path=resume_from_checkpoint)
    
    logging.info("✅ Production Stage 2 training completed successfully!")
    return trainer, training_module


def generate_sample_instruction_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate sample instruction tuning data."""
    import random
    
    data = []
    for i in range(num_samples):
        # Generate sample question-answer pairs
        questions = [
            "What is the time complexity of binary search?",
            "Explain the concept of recursion in programming.",
            "How do you implement a linked list in Python?",
            "What is the difference between a stack and a queue?",
            "How do you find the maximum element in an array?",
            "Explain the concept of object-oriented programming.",
            "What is the purpose of a hash table?",
            "How do you implement a binary tree?",
            "What is the difference between pass by value and pass by reference?",
            "Explain the concept of dynamic programming."
        ]
        
        reasonings = [
            "Binary search works by repeatedly dividing the search space in half. At each step, we compare the target with the middle element and eliminate half of the remaining elements.",
            "Recursion is a programming technique where a function calls itself to solve a problem. It consists of a base case and a recursive case.",
            "A linked list is a linear data structure where elements are stored in nodes, and each node contains a data field and a reference to the next node.",
            "A stack follows LIFO (Last In, First Out) principle, while a queue follows FIFO (First In, First Out) principle.",
            "To find the maximum element, we can iterate through the array once, keeping track of the largest element seen so far.",
            "Object-oriented programming is a programming paradigm based on the concept of objects, which contain data and code to manipulate that data.",
            "A hash table is a data structure that implements an associative array abstract data type, using a hash function to compute an index into an array of buckets.",
            "A binary tree is a tree data structure where each node has at most two children, referred to as the left child and the right child.",
            "Pass by value creates a copy of the argument, while pass by reference passes the actual memory address of the argument.",
            "Dynamic programming is a method for solving complex problems by breaking them down into simpler subproblems and storing the results of these subproblems."
        ]
        
        answers = [
            "The time complexity of binary search is O(log n), where n is the number of elements in the sorted array.",
            "Recursion is a powerful programming technique that allows a function to call itself. It's particularly useful for problems that can be broken down into smaller, similar subproblems.",
            "Here's a simple implementation: class Node: def __init__(self, data): self.data = data; self.next = None",
            "The key difference is in the order of element removal: stacks remove the most recently added element, while queues remove the least recently added element.",
            "The time complexity is O(n) where n is the array length. We need to examine each element at least once.",
            "OOP provides four main principles: encapsulation, inheritance, polymorphism, and abstraction, making code more modular and reusable.",
            "Hash tables provide average O(1) time complexity for search, insert, and delete operations, making them very efficient for many applications.",
            "Binary trees are useful for many algorithms including binary search trees, heaps, and expression trees.",
            "Pass by value is safer but can be slower for large objects, while pass by reference is more efficient but can lead to unintended side effects.",
            "Dynamic programming is particularly useful for optimization problems and can often reduce exponential time complexity to polynomial time."
        ]
        
        question = random.choice(questions)
        reasoning = random.choice(reasonings)
        answer = random.choice(answers)
        
        data.append({
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        })
    
    return data


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


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(description="Stage 2 ModularModel Training with Instruction Tuning")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="production",
                       choices=["basic", "production"],
                       help="Training mode: basic or production")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="config_production.yaml",
                       help="Config file path")
    
    # Common arguments
    parser.add_argument("--stage", type=str, default="stage2",
                       choices=["stage2"],
                       help="Training stage (must be stage2)")
    
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    
    parser.add_argument("--devices", type=int, default=None,
                       help="Number of devices")
    
    parser.add_argument("--precision", type=str, default=None,
                       choices=["16-mixed", "32", "bf16-mixed"],
                       help="Training precision")
    
    # Resume training arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
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
    parser.add_argument("--model_config", type=str, default="model_config_1.8B",
                       help="Model configuration key (production mode)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Log training mode selection
    logging.info("="*60)
    logging.info(f"🎯 SELECTED TRAINING MODE: {args.mode.upper()}")
    logging.info(f"🎯 STAGE: {args.stage.upper()}")
    logging.info("="*60)
    
    if args.mode == "basic":
        logging.info("📊 Training Method: PyTorch Lightning")
        logging.info("🔧 Framework: PyTorch Lightning + Instruction Tuning")
        logging.info("💡 Use case: Quick testing and development")
    elif args.mode == "production":
        logging.info("📊 Training Method: PyTorch Lightning")
        logging.info("🔧 Framework: PyTorch Lightning + Instruction Tuning")
        logging.info("💡 Use case: Production training with real datasets")
    
    logging.info("="*60)
    logging.info(f"🎯 Starting Stage 2 training in {args.mode} mode")
    
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
            # Use PyTorch Lightning backend
            train_kwargs = {
                "model_config_key": args.model_config,
                "stage": args.stage,
                "resume_from_checkpoint": args.resume_from_checkpoint,
            }
            
            # Only add arguments that are explicitly provided
            if args.devices is not None:
                train_kwargs["devices"] = args.devices
            if args.precision is not None:
                train_kwargs["precision"] = args.precision
            if args.learning_rate is not None:
                train_kwargs["learning_rate"] = args.learning_rate
            if args.batch_size is not None:
                train_kwargs["batch_size"] = args.batch_size
            
            trainer, module = train_production_mode(**train_kwargs)
        
        logging.info("🎉 Stage 2 training completed successfully!")
        logging.info("="*60)
        logging.info(f"✅ COMPLETED: {args.mode.upper()} MODE STAGE 2 TRAINING")
        logging.info("📊 Training Method Used: PyTorch Lightning")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"❌ Stage 2 training failed: {e}")
        raise


if __name__ == "__main__":
    main()

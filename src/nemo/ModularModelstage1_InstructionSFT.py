#!/usr/bin/env python3
"""
ModularModel Stage1 Instruction SFT Training Script for NeMo

This script provides instruction fine-tuning after Next Token Prediction (NTP) training.
It always loads a checkpoint from NTP training and fine-tunes on instruction datasets.

This is decoder-only training (stage1), similar to NTP but with instruction format data.
The full modular model (stage2) is not used yet.

Usage:
    python ModularModelstage1_InstructionSFT.py --mode production --model_config model_config_1.8B --stage stage1_inst_SFT
    python ModularModelstage1_InstructionSFT.py --mode production --model_config model_config_1.8B --stage stage1_inst_SFT --ntp_checkpoint_path outputs/checkpoints/stage1/last.ckpt
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

try:
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
    from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import IndexedDataset, MMapIndexedDataset
    from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler
    NEMO_DATASETS_AVAILABLE = True
except ImportError:
    NEMO_DATASETS_AVAILABLE = False
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

# Datasets import for instruction tuning
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Instruction dataset loading disabled.")
    load_dataset = None


def collate_fn(batch, tokenizer=None, max_length=2048):
    """
    Custom collate function to handle HuggingFace dataset batching with tokenization.
    Handles raw text data by tokenizing it on-the-fly.
    """
    if not batch:
        return {}
    
    # If batch is a list of dictionaries, process them
    if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
        # Check if we have pre-tokenized data (input_ids, attention_mask, labels)
        if 'input_ids' in batch[0] and 'attention_mask' in batch[0] and 'labels' in batch[0]:
            # Handle pre-tokenized data
            batched = {}
            # Only process training-relevant keys (skip 'id' field)
            training_keys = [key for key in batch[0].keys() if key in ['input_ids', 'attention_mask', 'labels', 'labels_shifted']]
            
            # Check if labels_shifted flag is already set in the data
            # If labels are pre-tokenized, they are likely already shifted
            # Handle both boolean and list formats (from batched tokenization)
            labels_shifted_val = batch[0].get('labels_shifted', True)
            if isinstance(labels_shifted_val, list):
                # If it's a list (from batched tokenization), check first value
                labels_shifted = labels_shifted_val[0] if len(labels_shifted_val) > 0 else True
            else:
                labels_shifted = labels_shifted_val if labels_shifted_val is not None else True
            
            # First, determine the target length for all sequences
            # Skip boolean values (like labels_shifted) when calculating lengths
            all_lengths = []
            for item in batch:
                for key in training_keys:
                    if key in item:
                        value = item[key]
                        # Skip boolean values and other non-sequence types
                        if isinstance(value, bool) or not hasattr(value, '__len__'):
                            continue
                        all_lengths.append(len(value))
            
            # Use the maximum length, but don't exceed max_length
            target_length = min(max(all_lengths), max_length)
            
            for key in training_keys:
                # Skip boolean values (like labels_shifted) - they don't need padding/truncation
                if key == 'labels_shifted':
                    # labels_shifted is a boolean flag, not a tensor
                    # We'll set it separately after processing other keys
                    continue
                
                tensors = [item[key] for item in batch]
                # Convert lists to tensors if needed
                for i, tensor in enumerate(tensors):
                    if isinstance(tensor, list):
                        tensors[i] = torch.tensor(tensor, dtype=torch.long)
                
                # Pad/truncate sequences to target_length
                padded_tensors = []
                for tensor in tensors:
                    if len(tensor) > target_length:
                        # Truncate if too long
                        padded_tensor = tensor[:target_length]
                    else:
                        # Pad if too short
                        pad_length = target_length - len(tensor)
                        if key == 'input_ids':
                            # Pad with 0 (pad token)
                            padded_tensor = torch.cat([tensor, torch.full((pad_length,), 0, dtype=torch.long)])
                        elif key == 'attention_mask':
                            # Pad attention mask with 0s
                            padded_tensor = torch.cat([tensor, torch.zeros(pad_length, dtype=torch.long)])
                        elif key == 'labels':
                            # Pad labels with -100 (ignore index)
                            padded_tensor = torch.cat([tensor, torch.full((pad_length,), -100, dtype=torch.long)])
                        else:
                            padded_tensor = torch.cat([tensor, torch.zeros(pad_length, dtype=torch.long)])
                    padded_tensors.append(padded_tensor)
                
                batched[key] = torch.stack(padded_tensors, dim=0)
            
            # Set labels_shifted flag (immutable once set)
            batched['labels_shifted'] = labels_shifted
            return batched
        # Check if we have text that needs tokenization
        elif 'text' in batch[0] and isinstance(batch[0]['text'], str):
            # Extract text from batch
            texts = [item.get('text', '') for item in batch]
            
            if tokenizer is not None:
                # Tokenize the texts
                tokenized = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # Create labels (shifted input_ids for next token prediction)
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                
                # SS debug: SHIFTING LABELS HERE (only if raw text, not pre-tokenized)
                # This path is for raw text that needs tokenization in collate_fn
                # If data comes from tokenize_function, labels are already shifted there
                # Create labels by shifting input_ids
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # Shift right: labels[i] = input_ids[i+1]
                labels[:, -1] = -100  # Ignore last token in loss calculation
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'labels_shifted': True  # Flag: labels are already shifted, do not shift again
                }
            else:
                # Fallback: try to stack existing tensors
                batched = {}
                # Preserve labels_shifted flag if it exists
                # CRITICAL: If labels exist, they are ALWAYS shifted (from tokenize_function or dataset)
                # Default to True for pre-tokenized data, False only if explicitly set to False
                labels_shifted_val = batch[0].get('labels_shifted', True)  # Changed default from False to True
                if isinstance(labels_shifted_val, list):
                    labels_shifted = labels_shifted_val[0] if len(labels_shifted_val) > 0 else True
                else:
                    labels_shifted = labels_shifted_val if labels_shifted_val is not None else True
                for key in batch[0].keys():
                    tensors = [item[key] for item in batch]
                    # Convert lists to tensors if needed
                    for i, tensor in enumerate(tensors):
                        if isinstance(tensor, list):
                            tensors[i] = torch.tensor(tensor, dtype=torch.long)
                    batched[key] = torch.stack(tensors, dim=0)
                # Preserve labels_shifted flag
                if 'labels' in batched:
                    batched['labels_shifted'] = labels_shifted
                return batched
        else:
            # Handle other pre-tokenized data formats
            batched = {}
            # Preserve labels_shifted flag if it exists
            # CRITICAL: If labels exist, they are ALWAYS shifted (from tokenize_function or dataset)
            # Default to True for pre-tokenized data, False only if explicitly set to False
            labels_shifted_val = batch[0].get('labels_shifted', True)  # Changed default from False to True
            if isinstance(labels_shifted_val, list):
                labels_shifted = labels_shifted_val[0] if len(labels_shifted_val) > 0 else True
            else:
                labels_shifted = labels_shifted_val if labels_shifted_val is not None else True
            for key in batch[0].keys():
                tensors = [item[key] for item in batch]
                # Convert lists to tensors if needed
                for i, tensor in enumerate(tensors):
                    if isinstance(tensor, list):
                        tensors[i] = torch.tensor(tensor, dtype=torch.long)
                batched[key] = torch.stack(tensors, dim=0)
            # Preserve labels_shifted flag
            if 'labels' in batched:
                batched['labels_shifted'] = labels_shifted
            return batched
    
    # If batch is already a dictionary, it means the default collate already processed it
    if isinstance(batch, dict):
        # Check if we have text that needs tokenization
        if 'text' in batch and isinstance(batch['text'], list):
            texts = batch['text']
            if tokenizer is not None:
                tokenized = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # SS debug: SHIFTING LABELS HERE (only if batch is dict with text list)
                # This is an alternative path for raw text tokenization
                # If data comes from tokenize_function, labels are already shifted there
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]  # Shift right: labels[i] = input_ids[i+1]
                labels[:, -1] = -100  # Ignore last token in loss calculation
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'labels_shifted': True  # Flag: labels are already shifted, do not shift again
                }
        
        # If already tokenized, return as is
        return batch
    
    # Fallback to default collate
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
        tokenizer=None,
    ):
        super().__init__()
        self.model = model
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Store tokenizer for debugging (optional)
        self.tokenizer = tokenizer
        
        # Counter for debug output (first 5 samples)
        self.debug_sample_count = 0
        self.max_debug_samples = 5
        
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
    
    def _debug_print_batch(self, batch, input_ids, labels, attention_mask, batch_idx, num_samples_to_print):
        """Print debug information for first 5 samples to verify training data."""
        try:
            # Get tokenizer if available
            tokenizer = self.tokenizer
            if tokenizer is None:
                # Try to get tokenizer from model if available
                if hasattr(self.model, 'tokenizer'):
                    tokenizer = self.model.tokenizer
                elif hasattr(self.model, 'modular_model') and hasattr(self.model.modular_model, 'tokenizer'):
                    tokenizer = self.model.modular_model.tokenizer
            
            # If still no tokenizer, try to load it
            if tokenizer is None:
                try:
                    from src.utils.tokenizer_manager import get_tokenizer_with_caching
                    tokenizer = get_tokenizer_with_caching(
                        tokenizer_path="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        custom_tokens=None,
                        force_download=False,
                        cache_dir="tokenizers"
                    )
                except Exception as e:
                    logging.warning(f"Could not load tokenizer for debugging: {e}")
                    tokenizer = None
            
            logging.info("=" * 80)
            logging.info(f"🔍 DEBUG: Training Batch {batch_idx}, Printing {num_samples_to_print} sample(s)")
            logging.info("=" * 80)
            
            labels_shifted = batch.get('labels_shifted', False)
            logging.info(f"📊 labels_shifted flag: {labels_shifted}")
            
            for sample_idx in range(num_samples_to_print):
                global_sample_num = self.debug_sample_count + sample_idx + 1
                logging.info(f"\n{'=' * 80}")
                logging.info(f"📝 SAMPLE {global_sample_num}/5 (batch_idx={batch_idx}, sample_idx={sample_idx})")
                logging.info(f"{'=' * 80}")
                
                # Get sample data
                sample_input_ids = input_ids[sample_idx].cpu().clone()
                sample_labels = labels[sample_idx].cpu().clone()
                sample_attention_mask = attention_mask[sample_idx].cpu().clone()
                
                # Remove padding (where attention_mask is 0)
                valid_length = sample_attention_mask.sum().item()
                sample_input_ids = sample_input_ids[:valid_length]
                sample_labels = sample_labels[:valid_length]
                sample_attention_mask = sample_attention_mask[:valid_length]
                
                # Print input_ids
                logging.info(f"\n📥 INPUT_IDS (shape: {sample_input_ids.shape}):")
                logging.info(f"   IDs: {sample_input_ids.tolist()[:50]}..." if len(sample_input_ids) > 50 else f"   IDs: {sample_input_ids.tolist()}")
                
                # Print attention_mask
                logging.info(f"\n🎭 ATTENTION_MASK (shape: {sample_attention_mask.shape}):")
                logging.info(f"   Mask: {sample_attention_mask.tolist()[:50]}..." if len(sample_attention_mask) > 50 else f"   Mask: {sample_attention_mask.tolist()}")
                
                # Print labels (token IDs)
                logging.info(f"\n🏷️  LABELS (token IDs, shape: {sample_labels.shape}):")
                logging.info(f"   IDs: {sample_labels.tolist()[:50]}..." if len(sample_labels) > 50 else f"   IDs: {sample_labels.tolist()}")
                
                # Decode to text if tokenizer available
                if tokenizer is not None:
                    try:
                        # Decode input_ids to text
                        input_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
                        logging.info(f"\n📝 INPUT_TEXT (decoded from input_ids):")
                        logging.info(f"   Text: {input_text[:200]}..." if len(input_text) > 200 else f"   Text: {input_text}")
                        
                        # Decode labels to text (filter out -100)
                        valid_label_ids = sample_labels[sample_labels != -100]
                        if len(valid_label_ids) > 0:
                            labels_text = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
                            logging.info(f"\n🏷️  LABELS_TEXT (decoded from labels, ignoring -100):")
                            logging.info(f"   Text: {labels_text[:200]}..." if len(labels_text) > 200 else f"   Text: {labels_text}")
                        else:
                            logging.info(f"\n🏷️  LABELS_TEXT: All labels are -100 (ignored)")
                        
                        # Verify label shifting
                        logging.info(f"\n✅ LABEL SHIFTING VERIFICATION:")
                        if labels_shifted:
                            # Labels are already shifted: labels[i] = input_ids[i+1]
                            # Check first few positions
                            num_check = min(10, len(sample_input_ids) - 1)
                            logging.info(f"   labels_shifted=True: labels[i] should equal input_ids[i+1]")
                            for i in range(num_check):
                                if sample_labels[i] != -100:
                                    expected = sample_input_ids[i + 1].item() if i + 1 < len(sample_input_ids) else -100
                                    actual = sample_labels[i].item()
                                    match = "✓" if actual == expected else "✗"
                                    logging.info(f"   Position {i}: label={actual}, expected (input_ids[{i+1}])={expected} {match}")
                        else:
                            # Labels are NOT shifted: labels[i] = input_ids[i]
                            # We need to shift them
                            logging.info(f"   labels_shifted=False: labels[i] should equal input_ids[i] (will shift in training_step)")
                            num_check = min(10, len(sample_input_ids))
                            for i in range(num_check):
                                if sample_labels[i] != -100:
                                    expected = sample_input_ids[i].item()
                                    actual = sample_labels[i].item()
                                    match = "✓" if actual == expected else "✗"
                                    logging.info(f"   Position {i}: label={actual}, expected (input_ids[{i}])={expected} {match}")
                        
                    except Exception as e:
                        logging.warning(f"Could not decode tokens to text: {e}")
                else:
                    logging.warning("Tokenizer not available - cannot decode to text")
                
                # Print shift_labels that will be used for loss
                logging.info(f"\n🔄 SHIFTED LABELS FOR LOSS CALCULATION:")
                if labels_shifted:
                    shift_labels = sample_labels[:-1]  # Slice labels[:-1]
                    logging.info(f"   labels_shifted=True: Using labels[:-1] (shape: {shift_labels.shape})")
                    logging.info(f"   Shift labels IDs: {shift_labels.tolist()[:30]}..." if len(shift_labels) > 30 else f"   Shift labels IDs: {shift_labels.tolist()}")
                else:
                    shift_labels = sample_labels[1:]  # Shift labels[1:]
                    logging.info(f"   labels_shifted=False: Using labels[1:] (shape: {shift_labels.shape})")
                    logging.info(f"   Shift labels IDs: {shift_labels.tolist()[:30]}..." if len(shift_labels) > 30 else f"   Shift labels IDs: {shift_labels.tolist()}")
                
                if tokenizer is not None and len(shift_labels) > 0:
                    try:
                        valid_shift_labels = shift_labels[shift_labels != -100]
                        if len(valid_shift_labels) > 0:
                            shift_labels_text = tokenizer.decode(valid_shift_labels, skip_special_tokens=False)
                            logging.info(f"   Shift labels text: {shift_labels_text[:200]}..." if len(shift_labels_text) > 200 else f"   Shift labels text: {shift_labels_text}")
                    except Exception as e:
                        logging.warning(f"Could not decode shift_labels: {e}")
            
            logging.info(f"\n{'=' * 80}\n")
            
        except Exception as e:
            logging.error(f"Error in debug print: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def training_step(self, batch, batch_idx):
        if self.stage == "stage1":
            # Stage 1: Next token prediction
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            
            # Debug output for first 5 samples (across all batches)
            if self.debug_sample_count < self.max_debug_samples:
                batch_size = input_ids.shape[0]
                samples_to_print = min(batch_size, self.max_debug_samples - self.debug_sample_count)
                self._debug_print_batch(batch, input_ids, labels, attention_mask, batch_idx, samples_to_print)
                self.debug_sample_count += samples_to_print
            
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
            # Check if labels are already shifted (set by dataset/collate function)
            # CRITICAL: If labels exist, they are ALWAYS shifted (from tokenize_function or dataset)
            # Default to True if labels exist, False only if explicitly set to False
            labels_shifted = batch.get('labels_shifted', True if 'labels' in batch else False)
            
            # SS debug: NOT SHIFTING AGAIN - just aligning dimensions
            # When labels_shifted=True: labels are already shifted (labels[i] = input_ids[i+1])
            # We only slice to align dimensions: logits[:-1] aligns with labels[:-1]
            # This is NOT a shift - it's just removing the last position to match logits
            if labels_shifted:
                # Labels are already shifted: labels[i] = input_ids[i+1]
                # We only need to shift logits to align with labels
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., :-1].contiguous()  # Slice labels[:-1] to align with logits[:-1] (NOT shifting)
            else:
                # Labels are NOT shifted: labels[i] = input_ids[i]
                # We need to shift both logits and labels
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()  # Shift labels to get next tokens
            
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
                # Use the same logic as loss calculation
                # CRITICAL: If labels exist, they are ALWAYS shifted (from tokenize_function or dataset)
                labels_shifted = batch.get('labels_shifted', True if 'labels' in batch else False)
                
                # SS debug: NOT SHIFTING AGAIN - just aligning dimensions for accuracy calculation
                # Same logic as loss calculation above
                if labels_shifted:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., :-1].contiguous()  # Slice, not shift
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()  # Shift labels
                
                predictions = torch.argmax(shift_logits, dim=-1)
                accuracy = (predictions == shift_labels).float().mean()
            else:
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
        
        # Log metrics with progress bar display
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log learning rate (this should make it appear in progress bar)
        try:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log("lr", current_lr, prog_bar=True, on_step=True)
        except RuntimeError:
            # Handle case when not attached to trainer (e.g., during testing)
            current_lr = self.learning_rate
            if hasattr(self, 'trainer') and self.trainer is not None:
                self.log("lr", current_lr, prog_bar=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Same as training step but with model.eval()
        self.model.eval()
        with torch.no_grad():
            if self.stage == "stage1":
                # Stage 1: Next token prediction
                # Debug: log batch keys
                logging.debug(f"Validation batch keys: {list(batch.keys())}")
                logging.debug(f"Validation batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")
                
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
                # Check if labels are already shifted (set by dataset/collate function)
                # CRITICAL: If labels exist, they are ALWAYS shifted (from tokenize_function or dataset)
                labels_shifted = batch.get('labels_shifted', True if 'labels' in batch else False)
                
                # SS debug: NOT SHIFTING AGAIN - just aligning dimensions (validation step)
                # Same logic as training_step above
                if labels_shifted:
                    # Labels are already shifted: labels[i] = input_ids[i+1]
                    # We only need to shift logits to align with labels
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., :-1].contiguous()  # Slice labels[:-1] to align (NOT shifting)
                else:
                    # Labels are NOT shifted: labels[i] = input_ids[i]
                    # We need to shift both logits and labels
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()  # Shift labels to get next tokens
                
                loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate validation metrics
                perplexity = torch.exp(loss)
                # Recalculate shift_logits and shift_labels for accuracy (same logic as loss)
                # CRITICAL: If labels exist, they are ALWAYS shifted (from tokenize_function or dataset)
                labels_shifted = batch.get('labels_shifted', True if 'labels' in batch else False)
                # SS debug: NOT SHIFTING AGAIN - just aligning dimensions for accuracy (validation)
                # Same logic as loss calculation above
                if labels_shifted:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., :-1].contiguous()  # Slice, not shift
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()  # Shift labels
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
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
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
            # Create a warmup scheduler first, then cosine annealing
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
        elif scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.scheduler_config.get("step_size", 1000)),
                gamma=float(self.scheduler_config.get("gamma", 0.1))
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
    
    # Load tokenizer with caching support
    from src.utils.tokenizer_manager import get_tokenizer_with_caching
    tokenizer = get_tokenizer_with_caching(
        tokenizer_path=tokenizer_path,
        custom_tokens=None,  # Use default special tokens
        force_download=False,
        cache_dir="tokenizers"
    )
    
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
        tokenizer_path = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
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
    tokenizer_path = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    if base_path is not None:
        tokenizer_path = str(Path(base_path) / tokenizer_path)
    
    loader = HuggingFaceDatasetLoader(config_path, tokenizer_path, stage="stage1")
    training_data = loader.create_training_data(total_samples=total_samples)
    
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


def setup_config_file(config_path: str):
    """Setup configuration from specified config file path."""
    import shutil
    
    # Handle different input formats
    if config_path == "basic" or config_path == "config.yaml":
        # Use existing config.yaml without copying
        logging.info("✅ Using existing config.yaml configuration!")
        return
    
    # Determine the actual config file path
    if config_path.startswith("configs/"):
        # Full path provided
        config_file = config_path
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        # Just filename provided
        config_file = f"configs/{config_path}"
    else:
        # Assume it's a config name without extension
        config_file = f"configs/{config_path}.yaml"
    
    # Extract config name for logging
    config_name = os.path.basename(config_file).replace('.yaml', '').replace('.yml', '')
    
    logging.info(f"🚀 Setting up {config_name} configuration...")
    
    # Check if config exists
    if not os.path.exists(config_file):
        logging.error(f"❌ Config file not found: {config_file}")
        logging.error(f"💡 Make sure the file exists or provide the correct path")
        return
    
    # Copy config to active config
    logging.info(f"📝 Copying {config_name} config to active config...")
    shutil.copy(config_file, 'configs/config.yaml')
    logging.info(f"✅ {config_name.capitalize()} configuration ready!")


def check_checkpoint_compatibility(checkpoint_path: str, training_module, logger):
    """
    Check if checkpoint is compatible with current model structure.
    Returns True if compatible, False if incompatible.
    """
    try:
        # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain tokenizer objects
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' not in checkpoint:
            return False
            
        state_dict = checkpoint['state_dict']
        
        # Check for weight tying conflicts
        has_embed_tokens = any('embed_tokens.weight' in key for key in state_dict.keys())
        has_lm_head = any('lm_head.weight' in key for key in state_dict.keys())
        
        # Get current model structure
        current_state_dict = training_module.state_dict()
        current_has_embed_tokens = any('embed_tokens.weight' in key for key in current_state_dict.keys())
        current_has_lm_head = any('lm_head.weight' in key for key in current_state_dict.keys())
        
        # Check if weight tying status matches
        checkpoint_weight_tying = has_embed_tokens and has_lm_head and len([k for k in state_dict.keys() if 'lm_head.weight' in k]) == 1
        current_weight_tying = current_has_embed_tokens and current_has_lm_head and len([k for k in current_state_dict.keys() if 'lm_head.weight' in k]) == 1
        
        if checkpoint_weight_tying != current_weight_tying:
            logger.warning(f"⚠️ Weight tying mismatch: checkpoint={checkpoint_weight_tying}, current={current_weight_tying}")
            return False
        
        # Check optimizer state compatibility
        # Get the current model's trainable parameters to compare with optimizer state
        current_params = list(training_module.model.parameters())
        current_trainable_params = [p for p in current_params if p.requires_grad]
        current_param_count = len(current_trainable_params)
        
        # Also count total parameters (including frozen) for comparison
        current_total_params = len(current_params)
        
        # Check checkpoint optimizer state
        if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0:
            checkpoint_optimizer = checkpoint['optimizer_states'][0]
            checkpoint_param_count = 0
            
            # Count unique parameters in optimizer state
            if 'state' in checkpoint_optimizer:
                checkpoint_param_count = len(checkpoint_optimizer['state'])
            
            # Get parameter group info
            checkpoint_param_groups = checkpoint_optimizer.get('param_groups', [])
            checkpoint_total_params_in_groups = sum(len(g.get('params', [])) for g in checkpoint_param_groups)
            
            # More lenient compatibility check:
            # PyTorch Lightning can handle optimizer state mismatches by filtering
            # We'll be lenient and let PyTorch Lightning try to load, then fall back if needed
            # 1. If trainable params match exactly -> compatible
            # 2. If checkpoint has MORE params -> might be compatible (params frozen, PyTorch Lightning will filter)
            # 3. If checkpoint has FEWER params -> incompatible (model grew, can't restore optimizer state)
            
            if checkpoint_total_params_in_groups == current_param_count:
                # Exact match - fully compatible
                logger.info(f"✅ Optimizer state compatible: {checkpoint_total_params_in_groups} params match")
                return True
            elif checkpoint_total_params_in_groups > current_param_count:
                # Checkpoint has more params - likely due to parameter freezing
                # PyTorch Lightning can filter optimizer state for frozen params
                logger.info(f"ℹ️  Optimizer state has more params ({checkpoint_total_params_in_groups}) than trainable ({current_param_count})")
                logger.info(f"   This is likely due to parameter freezing. PyTorch Lightning will filter optimizer state.")
                logger.info(f"   Frozen params: {current_total_params - current_param_count}")
                logger.info(f"   Will attempt to load optimizer state (PyTorch Lightning will handle filtering)")
                # Allow loading - PyTorch Lightning will filter out optimizer state for frozen params
                return True
            else:
                # Checkpoint has fewer params - model grew, incompatible
                logger.warning(f"⚠️ Optimizer state mismatch:")
                logger.warning(f"   - Checkpoint has {checkpoint_total_params_in_groups} params")
                logger.warning(f"   - Current model has {current_param_count} trainable params")
                logger.warning(f"   - Checkpoint has fewer params than current model (model architecture changed)")
                logger.warning("Will load model weights only and restart optimizer from scratch")
                return False
            
        return True
        
    except Exception as e:
        logger.warning(f"❌ Error checking checkpoint compatibility: {e}")
        return False


def load_checkpoint_with_fallback(checkpoint_path: str, training_module, logger):
    """
    Load checkpoint with fallback mechanisms to handle parameter mapping issues.
    
    This function handles the case where checkpoint was saved with weight tying enabled
    but current model has weight tying disabled (or vice versa).
    """
    try:
        logger.info(f"🔄 Attempting checkpoint loading from: {checkpoint_path}")
        # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain tokenizer objects
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Create a filtered state dict that handles weight tying mismatches
            filtered_state_dict = {}
            embed_tokens_weight = None
            
            for key, value in state_dict.items():
                # Skip optimizer and scheduler states
                if key.startswith('optimizer') or key.startswith('lr_schedulers'):
                    continue
                
                # Handle weight tying conflicts
                if 'embed_tokens.weight' in key:
                    # Store the embedding weight for potential use with LM head
                    embed_tokens_weight = value
                    filtered_state_dict[key] = value
                    logger.info(f"📝 Stored embedding weight: {key}")
                    
                elif 'lm_head.weight' in key:
                    # Check if we have a separate LM head (no weight tying)
                    if 'model.decoder.lm_head.weight' in filtered_state_dict:
                        # We already have an LM head weight, skip this one
                        logger.info(f"⏭️ Skipping duplicate LM head weight: {key}")
                        continue
                    else:
                        # Use the LM head weight from checkpoint
                        filtered_state_dict[key] = value
                        logger.info(f"📝 Using LM head weight from checkpoint: {key}")
                        
                else:
                    # All other parameters
                    filtered_state_dict[key] = value
            
            # If we have embedding weight but no LM head weight, and current model doesn't use weight tying,
            # we need to initialize the LM head separately
            if embed_tokens_weight is not None and 'model.decoder.lm_head.weight' not in filtered_state_dict:
                logger.info("🔧 Initializing LM head weight separately (no weight tying)")
                # The LM head will be initialized randomly, which is fine for training continuation
            
            # Load with strict=False to handle parameter mismatches
            missing_keys, unexpected_keys = training_module.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                logger.info(f"⚠️ Missing keys (will use random initialization): {len(missing_keys)}")
                if logger.level <= logging.DEBUG:
                    logger.debug(f"Missing keys: {missing_keys[:10]}...")  # Show first 10
            
            if unexpected_keys:
                logger.info(f"⚠️ Unexpected keys (ignored): {len(unexpected_keys)}")
                if logger.level <= logging.DEBUG:
                    logger.debug(f"Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
            
            logger.info("✅ Successfully loaded checkpoint with weight tying compatibility")
            return True
            
        else:
            logger.warning("⚠️ Checkpoint does not contain 'state_dict' key")
            return False
            
    except Exception as e:
        logger.warning(f"❌ Failed to load checkpoint: {e}")
        return False


def create_strategy(distributed_config: dict):
    """Create the appropriate training strategy based on configuration."""
    if not LIGHTNING_AVAILABLE:
        return None
    
    strategy_name = distributed_config.get("strategy", "auto")
    fsdp_config = distributed_config.get("fsdp", {})
    devices = distributed_config.get("devices", "auto")
    
    # Check if we're using single GPU
    is_single_gpu = False
    if isinstance(devices, int) and devices == 1:
        is_single_gpu = True
    elif devices == "auto":
        # Check if we're actually using single GPU by checking available devices
        import torch
        is_single_gpu = torch.cuda.device_count() == 1
    
    # For single GPU, avoid FSDP due to parameter mapping issues
    if is_single_gpu and (strategy_name == "fsdp" or (strategy_name == "auto" and fsdp_config.get("enabled", False))):
        logging.warning("⚠️ Single GPU detected - disabling FSDP to avoid parameter mapping issues")
        logging.info("Using DDP strategy with find_unused_parameters=True for single GPU training")
        return "ddp_find_unused_parameters_true"
    
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
        
        # Handle activation checkpointing with new policy format
        activation_checkpointing = fsdp_config.get("activation_checkpointing", True)
        if activation_checkpointing:
            # Import common transformer layer classes for activation checkpointing
            import torch.nn as nn
            # Use new policy format for activation checkpointing
            # Apply checkpointing to common large layers that benefit from it
            activation_checkpointing_policy = {
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
                nn.MultiheadAttention,
                nn.Linear,  # Large linear layers
            }
        else:
            activation_checkpointing_policy = None
        
        strategy = FSDPStrategy(
            cpu_offload=fsdp_config.get("cpu_offload", False),
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
            limit_all_gathers=fsdp_config.get("limit_all_gathers", True),
            activation_checkpointing_policy=activation_checkpointing_policy,
            use_orig_params=fsdp_config.get("use_orig_params", False),
        )
        
        logging.info(f"Created FSDP strategy with sharding_strategy={sharding_strategy}, "
                    f"cpu_offload={fsdp_config.get('cpu_offload', False)}")
        return strategy
    
    elif strategy_name == "ddp":
        # Create DDP strategy with find_unused_parameters=True
        logging.info("Using DDP strategy with find_unused_parameters=True")
        return "ddp_find_unused_parameters_true"
    
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
            config = create_nemo_config_from_existing("model_config_243M", stage)
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
        equal_percentage = 100.0 / len(pretraining_datasets) if pretraining_datasets else 100.0
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
                
            logging.info(f"Loading instruction dataset: {dataset_name} ({percentage}% = {dataset_samples} samples)")
            
            # Handle different split names for different datasets
            # Some datasets use "train_sft" instead of "train"
            actual_split = split
            if split == "train":
                # Try train_sft first for datasets like OpenHermes-2.5-H4
                try:
                    if subset:
                        dataset = load_dataset(dataset_name, subset, split="train_sft")
                    else:
                        dataset = load_dataset(dataset_name, split="train_sft")
                    actual_split = "train_sft"
                    logging.info(f"  Using split 'train_sft' for {dataset_name}")
                except (ValueError, KeyError):
                    # Fall back to regular train split
                    if subset:
                        dataset = load_dataset(dataset_name, subset, split=split)
                    else:
                        dataset = load_dataset(dataset_name, split=split)
            elif split == "validation":
                # Try validation splits with different names
                try:
                    if subset:
                        dataset = load_dataset(dataset_name, subset, split="test_sft")
                    else:
                        dataset = load_dataset(dataset_name, split="test_sft")
                    actual_split = "test_sft"
                    logging.info(f"  Using split 'test_sft' for {dataset_name}")
                except (ValueError, KeyError):
                    try:
                        if subset:
                            dataset = load_dataset(dataset_name, subset, split="validation")
                        else:
                            dataset = load_dataset(dataset_name, split="validation")
                    except (ValueError, KeyError):
                        # Fall back to test split
                        if subset:
                            dataset = load_dataset(dataset_name, subset, split="test")
                        else:
                            dataset = load_dataset(dataset_name, split="test")
            else:
                # Load dataset with optional subset
                if subset:
                    dataset = load_dataset(dataset_name, subset, split=split)
                else:
                    dataset = load_dataset(dataset_name, split=split)
            
            # Convert to our format
            dataset_data = []
            for i, item in enumerate(dataset):
                if i >= dataset_samples:
                    break
                
                # Handle different dataset formats
                if "messages" in item:
                    # Messages format (tulu-v2-sft-mixture, OpenHermes-2.5-H4, etc.)
                    # Extract user and assistant messages from conversation
                    messages = item["messages"]
                    if not isinstance(messages, list) or len(messages) < 2:
                        continue
                    
                    # Find user and assistant messages
                    user_content = None
                    assistant_content = None
                    
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role", "").lower()
                            content = msg.get("content", "")
                            if role == "user" and user_content is None:
                                user_content = content
                            elif role in ["assistant", "gpt"] and assistant_content is None:
                                assistant_content = content
                    
                    if user_content and assistant_content:
                        sample = {
                            "question": str(user_content),
                            "answer": str(assistant_content),
                            "dataset": dataset_name
                        }
                    else:
                        # Skip if we can't extract user/assistant pair
                        continue
                        
                elif "problem" in item and "expected_answer" in item:
                    # nvidia/OpenMathReasoning format
                    sample = {
                        "question": item["problem"],
                        "answer": item["expected_answer"],
                        "dataset": dataset_name
                    }
                elif "question" in item and "answer" in item:
                    # Generic question-answer format
                    sample = {
                        "question": item["question"],
                        "answer": item["answer"],
                        "dataset": dataset_name
                    }
                elif "input" in item and "output" in item:
                    # Generic instruction format
                    sample = {
                        "question": item["input"],
                        "answer": item["output"],
                        "dataset": dataset_name
                    }
                elif "instruction" in item and "response" in item:
                    # Instruction-response format (some datasets use this)
                    sample = {
                        "question": item["instruction"],
                        "answer": item["response"],
                        "dataset": dataset_name
                    }
                elif "prompt" in item and "completion" in item:
                    # Prompt-completion format
                    sample = {
                        "question": item["prompt"],
                        "answer": item["completion"],
                        "dataset": dataset_name
                    }
                else:
                    # Log unknown format for debugging (first time only per dataset)
                    if i == 0:
                        logging.warning(f"Unknown format in {dataset_name}: {list(item.keys())}. Skipping this dataset.")
                    continue
                
                dataset_data.append(sample)
            
            all_data.extend(dataset_data)
            logging.info(f"Loaded {len(dataset_data)} samples from {dataset_name}")
            
        except Exception as e:
            logging.warning(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    logging.info(f"Total instruction samples loaded: {len(all_data)}")
    return all_data


def find_ntp_checkpoint(ntp_checkpoint_path: Optional[str] = None, config: Dict[str, Any] = None) -> str:
    """
    Find the NTP training checkpoint to load.
    This is REQUIRED for instruction SFT - will raise error if not found.
    Priority: explicit path > auto-detect from NTP checkpoint dir > config checkpoint dir
    
    Returns:
        str: Path to NTP checkpoint (never None - raises error if not found)
    
    Raises:
        FileNotFoundError: If no NTP checkpoint is found
    """
    if ntp_checkpoint_path:
        if os.path.exists(ntp_checkpoint_path):
            logging.info(f"✅ Using provided NTP checkpoint: {ntp_checkpoint_path}")
            return ntp_checkpoint_path
        else:
            logging.error(f"❌ Provided NTP checkpoint path does not exist: {ntp_checkpoint_path}")
            raise FileNotFoundError(f"NTP checkpoint not found at provided path: {ntp_checkpoint_path}")
    
    # Try to find NTP checkpoint directory
    ntp_checkpoint_dir = None
    if config:
        # Try stage1 checkpoint dir (where NTP training saves checkpoints)
        ntp_checkpoint_dir = config.get("checkpoint_dir", "outputs/checkpoints/stage1")
        # If checkpoint_dir is generic, try stage1 subdirectory
        if not os.path.exists(ntp_checkpoint_dir):
            ntp_checkpoint_dir = "outputs/checkpoints/stage1"
    
    if ntp_checkpoint_dir and os.path.exists(ntp_checkpoint_dir):
        # Look for latest checkpoint
        checkpoint_dir = Path(ntp_checkpoint_dir)
        last_checkpoint = checkpoint_dir / "last.ckpt"
        if last_checkpoint.exists():
            logging.info(f"✅ Auto-detected NTP checkpoint: {last_checkpoint}")
            return str(last_checkpoint)
        
        # Look for any checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            logging.info(f"✅ Auto-detected latest NTP checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)
    
    # CRITICAL: Instruction SFT REQUIRES NTP checkpoint - fail fast
    logging.error("=" * 80)
    logging.error("❌ CRITICAL ERROR: NTP checkpoint is REQUIRED for instruction SFT!")
    logging.error("=" * 80)
    logging.error("Instruction SFT cannot start from a newly initialized model.")
    logging.error("You must first train the model with Next Token Prediction (NTP) training.")
    logging.error("")
    logging.error("Please provide one of the following:")
    logging.error("  1. --ntp_checkpoint_path <path_to_ntp_checkpoint.ckpt>")
    logging.error("  2. Ensure NTP checkpoints exist in: outputs/checkpoints/stage1/")
    logging.error("")
    logging.error("To train with NTP first, run:")
    logging.error("  python ModularModelstage1_NTPtraining.py --mode production --model_config <config> --stage stage1")
    logging.error("=" * 80)
    raise FileNotFoundError(
        "NTP checkpoint is REQUIRED for instruction SFT. "
        "Instruction SFT cannot start from a newly initialized model. "
        "Please provide --ntp_checkpoint_path or ensure NTP checkpoints exist in outputs/checkpoints/stage1/"
    )


def train_production_mode(
    model_config_key: str = "model_config_1.8B",
    stage: str = "stage1_inst_SFT",
    devices: int = 1,
    precision: str = "bf16-mixed",
    base_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    ntp_checkpoint_path: Optional[str] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    benchmark: bool = True,
    **kwargs
):
    """Production training mode using configuration files for instruction SFT."""
    
    if not LIGHTNING_AVAILABLE:
        raise RuntimeError("PyTorch Lightning not available for production training mode.")
    
    if create_nemo_config_from_existing is None:
        raise RuntimeError("Config loader not available for production training mode.")
    
    # Ensure we're using the correct stage for instruction SFT
    if stage != "stage1_inst_SFT":
        logging.warning(f"⚠️ Stage '{stage}' specified, but instruction SFT requires 'stage1_inst_SFT'. Using 'stage1_inst_SFT' instead.")
        stage = "stage1_inst_SFT"
    
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
    
    logging.info(f"🚀 Starting instruction SFT training: {model_config_key} - {stage}")
    logging.info(f"📊 Training Method: PyTorch Lightning")
    logging.info(f"🔧 Framework: PyTorch Lightning + Instruction Datasets")
    logging.info(f"📈 Progress Bar: PyTorch Lightning Default (RichProgressBar disabled due to compatibility issues)")
    logging.info(f"🎯 Instruction SFT: REQUIRES checkpoint from NTP training (cannot start from scratch)")
    
    # CRITICAL: Find NTP checkpoint FIRST before loading config or creating model
    # This ensures we fail fast if NTP checkpoint is missing
    logging.info("=" * 80)
    logging.info("🔍 STEP 1: Locating NTP checkpoint (REQUIRED for instruction SFT)...")
    logging.info("=" * 80)
    
    # Load minimal config first just to get checkpoint directory
    # Use stage1_inst_SFT for training config, but model will be created as stage1 (decoder-only)
    temp_config = create_nemo_config_from_existing(model_config_key, stage, base_path)
    
    # Find NTP checkpoint - this will raise FileNotFoundError if not found
    try:
        ntp_checkpoint = find_ntp_checkpoint(ntp_checkpoint_path, temp_config)
        logging.info(f"✅ NTP checkpoint found: {ntp_checkpoint}")
        logging.info("=" * 80)
    except FileNotFoundError as e:
        # Re-raise with clear error message
        raise FileNotFoundError(
            f"Instruction SFT requires an NTP checkpoint. {str(e)}"
        ) from e
    
    # Now load full configuration
    config = temp_config
    
    # Override parameters if provided (but don't override with None values or defaults)
    # Only override if the value is explicitly provided and different from config
    for key, value in kwargs.items():
        if key in config and value is not None:
            # Log when overriding config values
            if config[key] != value:
                logging.info(f"🔄 Overriding config {key}: {config[key]} → {value}")
            config[key] = value
    
    # Set default values for missing keys (only if not provided via command line)
    if "devices" not in config:
        config["devices"] = devices if devices is not None else 1
    if "precision" not in config:
        config["precision"] = precision if precision is not None else "bf16-mixed"
    if "batch_size" not in config:
        batch_size_arg = kwargs.get("batch_size")
        config["batch_size"] = batch_size_arg if batch_size_arg is not None else 4
    if "learning_rate" not in config:
        lr_arg = kwargs.get("learning_rate")
        config["learning_rate"] = lr_arg if lr_arg is not None else 1e-5
    if "checkpoint_dir" not in config:
        config["checkpoint_dir"] = "outputs/checkpoints"
    
    # Log final configuration values being used
    logging.info(f"🎯 Final Configuration Values:")
    logging.info(f"   batch_size: {config.get('batch_size')}")
    logging.info(f"   learning_rate: {config.get('learning_rate')}")
    logging.info(f"   devices: {config.get('devices')}")
    logging.info(f"   precision: {config.get('precision')}")
    logging.info(f"   sequence_length: {config.get('sequence_length')}")
    logging.info(f"   gradient_accumulation_steps: {config.get('gradient_accumulation_steps')}")
    logging.info(f"   num_workers: {config.get('num_workers')}")
    logging.info(f"   pin_memory: {config.get('pin_memory')}")
    logging.info(f"   persistent_workers: {config.get('persistent_workers')}")
    logging.info(f"   prefetch_factor: {config.get('prefetch_factor')}")
    
    # Create model - use stage1 for model architecture (decoder-only)
    # The training config comes from stage1_inst_SFT, but model is decoder-only (stage1)
    config["training_stage"] = "stage1"  # Model is decoder-only (stage1), not full modular model
    model = create_modular_model_nemo(**config)
    
    # Load instruction datasets for SFT
    vocab_size = config["vocab_size"]
    tokenizer_path = config.get("tokenizer_path", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    
    # Load instruction tuning datasets from config
    data_config = config.get("data", {})
    pretraining_datasets = data_config.get("pretraining_datasets", {})
    
    # Extract dataset names from config (fallback to default if not found)
    if not pretraining_datasets:
        logging.warning("No instruction datasets found in config. Using default instruction datasets.")
        pretraining_datasets = {
            "nvidia/OpenCodeReasoning": {"percentage": 50},
            "nvidia/OpenMathReasoning": {"percentage": 50}
        }
    
    max_samples = data_config.get("processing", {}).get("total_samples", 100000)
    
    logging.info(f"📚 Loading instruction tuning datasets from config")
    logging.info(f"   Datasets: {list(pretraining_datasets.keys())}")
    logging.info(f"   Max samples: {max_samples}")
    
    # Load instruction datasets
    train_instruction_data = load_instruction_datasets_with_percentages(
        pretraining_datasets, 
        max_samples, 
        "train"
    )
    val_instruction_data = load_instruction_datasets_with_percentages(
        pretraining_datasets, 
        max(max_samples // 10, 100),  # At least 100 validation samples
        "validation"
    )
    
    logging.info(f"✅ Loaded {len(train_instruction_data)} training and {len(val_instruction_data)} validation instruction samples")
    
    # Convert instruction data to HuggingFace dataset format for tokenization
    from datasets import Dataset as HFDataset
    train_hf_dataset = HFDataset.from_list(train_instruction_data)
    val_hf_dataset = HFDataset.from_list(val_instruction_data)
    
    # Combine train and val for processing, then split later
    from datasets import concatenate_datasets
    hf_dataset = concatenate_datasets([train_hf_dataset, val_hf_dataset])
    
    # Load tokenizer with caching support
    from src.utils.tokenizer_manager import get_tokenizer_with_caching
    tokenizer = get_tokenizer_with_caching(
        tokenizer_path=tokenizer_path,
        custom_tokens=None,  # Use default special tokens
        force_download=False,
        cache_dir="tokenizers"
    )
    
    # Get sequence length from config
    seq_length = config.get("sequence_length", 2048)
    
    # Special tokens for instruction format (simpler than stage2 - no reasoning token)
    instruction_token = "<|instruction|>"
    response_token = "<|response|>"
    end_token = "<|end|>"
    
    # Define tokenization function for instruction format
    def tokenize_instruction_function(examples):
        """Tokenize instruction examples with proper masking for instruction SFT."""
        batch_size = len(examples['question'])
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for i in range(batch_size):
            question = str(examples['question'][i]) if examples['question'][i] else ""
            answer = str(examples['answer'][i]) if examples['answer'][i] else ""
            
            # Format: <|instruction|> {question} <|response|> {answer} <|end|>
            instruction_text = f"{instruction_token} {question} {response_token} {answer} {end_token}"
            
            # Tokenize the full instruction
            tokenized = tokenizer(
                instruction_text,
                padding=False,
                truncation=True,
                max_length=seq_length,
                return_tensors=None
            )
            
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            # Find where response starts (after response_token)
            response_token_id = tokenizer.convert_tokens_to_ids(response_token)
            try:
                response_start_idx = input_ids.index(response_token_id) + 1  # +1 to start after token
            except ValueError:
                # If response_token not found, assume response starts after instruction_token
                instruction_token_id = tokenizer.convert_tokens_to_ids(instruction_token)
                try:
                    response_start_idx = input_ids.index(instruction_token_id) + len(tokenizer.encode(question)) + 1
                except ValueError:
                    # Fallback: mask everything except last 20% (assumed to be response)
                    response_start_idx = int(len(input_ids) * 0.8)
            
            # Create labels: mask instruction tokens (-100), keep response tokens
            labels = [-100] * len(input_ids)  # Start with all masked
            # Only compute loss on response tokens (after response_start_idx)
            # Shift labels for next token prediction: labels[i] = input_ids[i+1]
            for j in range(response_start_idx, len(input_ids) - 1):
                labels[j] = input_ids[j + 1]
            labels[-1] = -100  # Last token always ignored
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list,
            'labels_shifted': [True] * batch_size  # Labels are already shifted
        }
    
    # Tokenize the instruction dataset
    logging.info("🔄 Tokenizing instruction dataset...")
    logging.info(f"   - Total samples: {len(hf_dataset)}")
    
    tokenization_workers = min(48, config.get('num_workers', 8) * 4)
    logging.info(f"   - Using {tokenization_workers} workers for tokenization")
    
    hf_dataset = hf_dataset.map(
        tokenize_instruction_function,
        batched=True,
        batch_size=2000,
        num_proc=tokenization_workers,
        remove_columns=['question', 'answer', 'dataset'] if 'dataset' in hf_dataset.column_names else ['question', 'answer'],
        desc='Tokenizing instruction dataset'
    )
    
    logging.info("✅ Instruction dataset tokenization completed!")
    logging.info(f"📊 Dataset features: {list(hf_dataset.features.keys())}")
    
    # Split back into train and val
    train_size = len(train_instruction_data)
    train_hf_dataset = hf_dataset.select(range(train_size))
    val_hf_dataset = hf_dataset.select(range(train_size, train_size + len(val_instruction_data)))
    
    is_tokenized = True
    
    # Create PyTorch Lightning DataModule with HuggingFace dataset
    from lightning.pytorch import LightningDataModule
    from torch.utils.data import DataLoader
    
    # Load tokenizer if not already loaded
    if 'tokenizer' not in locals():
        from src.utils.tokenizer_manager import get_tokenizer_with_caching
        tokenizer = get_tokenizer_with_caching(
            tokenizer_path=tokenizer_path,
            custom_tokens=None,  # Use default special tokens
            force_download=False,
            cache_dir="tokenizers"
        )

    # Create the custom DataModule with train/val datasets
    # We need to create separate datasets for train and val
    class InstructionDataModule(LightningDataModule):
        def __init__(self, train_dataset, val_dataset, tokenizer, batch_size=8, seq_length=2048, num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=None, seed=None):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.persistent_workers = persistent_workers
            self.collate_fn = collate_fn
            
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)
        
        def train_dataloader(self):
            def simple_collate_fn(batch):
                return collate_fn(batch, tokenizer=None, max_length=self.seq_length)
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                collate_fn=simple_collate_fn,
                generator=self.generator
            )
        
        def val_dataloader(self):
            def simple_collate_fn(batch):
                return collate_fn(batch, tokenizer=None, max_length=self.seq_length)
            
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                collate_fn=simple_collate_fn
            )
    
    # Create the custom DataModule
    data_module = InstructionDataModule(
        train_dataset=train_hf_dataset,
        val_dataset=val_hf_dataset,
        tokenizer=tokenizer,
        batch_size=config.get("batch_size", 8),
        seq_length=config.get("sequence_length", 2048),
        num_workers=config.get("num_workers", 8),
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True),
        collate_fn=collate_fn,
        seed=seed
    )

    # Get the data loaders from the custom data module
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logging.info("✅ Created instruction DataModule with fully tokenized datasets")
    logging.info("🚀 Instruction datasets are completely tokenized - maximum training performance!")
    logging.info("⚡ No tokenization overhead during training - optimal GPU utilization!")
    logging.info("🎯 Loss computed only on response tokens (instruction tokens masked)")
    
    # Create training module with optimizer and scheduler configs
    # Pass tokenizer for debugging (first 5 samples)
    training_module = ModularModelTrainingModule(
        model=model,
        stage=stage,
        learning_rate=config["learning_rate"],
        optimizer_config=config.get("optimizer", {}),
        scheduler_config=config.get("scheduler", {}),
        tokenizer=tokenizer  # Pass tokenizer for debug output
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
    
    # Handle validation interval (0 means disable validation)
    if val_check_interval_steps == 0:
        # Disable validation by setting to a very large number
        val_check_interval_steps = float('inf')
        logging.info("Validation disabled (val_check_interval_steps = 0)")
    elif val_check_interval_steps > total_steps:
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
    if val_check_interval_steps == float('inf'):
        logging.info(f"   - Validation: DISABLED")
    else:
        logging.info(f"   - Validate every {val_check_interval_steps} steps")
    logging.info(f"   - Gradient clipping: {config.get('gradient_clip_val', 1.0)} ({config.get('gradient_clip_algorithm', 'norm')})")
    logging.info(f"   - Checkpoint directory: {config['checkpoint_dir']}")
    logging.info(f"   - Best checkpoints: checkpoint-{{step:06d}}-{{val_loss:.4f}}.ckpt (top {checkpointing_config.get('save_top_k', 3)})")
    logging.info(f"   - Latest checkpoints: last-checkpoint-{{step:06d}}.ckpt")
    
    # Use step-based checkpointing (direct step intervals)
    step_checkpoint = ModelCheckpoint(
        dirpath=config["checkpoint_dir"],
        filename="checkpoint-{step:06d}",  # Removed val_loss to avoid type errors when val hasn't run
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
        auto_insert_metric_name=False,
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
    # Create Lightning profiler for performance analysis
    from lightning.pytorch.profilers import PyTorchProfiler
    profiler = PyTorchProfiler(
        dirpath="./profiler_logs",
        filename="training_profile",
        export_to_chrome=True,
        row_limit=100,
        sort_by_key="cuda_time_total",
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs')
    )
    
    trainer_kwargs = {
        "devices": devices,
        "num_nodes": num_nodes,
        "precision": config["precision"],
        "strategy": strategy,
        "callbacks": callbacks,
        "logger": logger,
        "profiler": profiler,
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
    
    # For instruction SFT, we ALWAYS load from NTP checkpoint first
    # Then optionally resume from instruction SFT checkpoint if provided
    initial_checkpoint_path = ntp_checkpoint  # Always start from NTP checkpoint
    
    # Check if we should resume from instruction SFT checkpoint
    instruction_sft_checkpoint = resume_from_checkpoint
    if instruction_sft_checkpoint:
        logging.info(f"🔄 Will resume instruction SFT from: {instruction_sft_checkpoint}")
        # Load NTP checkpoint first, then resume from instruction SFT checkpoint
        checkpoint_path = instruction_sft_checkpoint
    else:
        # Start fresh instruction SFT from NTP checkpoint
        checkpoint_path = initial_checkpoint_path
        logging.info(f"🔄 Starting instruction SFT from NTP checkpoint: {checkpoint_path}")
    
    # CRITICAL: Load NTP checkpoint (always required for instruction SFT)
    # This ensures we never start from a newly initialized model
    logging.info("=" * 80)
    logging.info(f"📥 STEP 2: Loading NTP checkpoint (REQUIRED for instruction SFT)...")
    logging.info(f"   Checkpoint: {initial_checkpoint_path}")
    logging.info("=" * 80)
    
    if not os.path.exists(initial_checkpoint_path):
        logging.error(f"❌ NTP checkpoint file does not exist: {initial_checkpoint_path}")
        raise FileNotFoundError(
            f"NTP checkpoint file not found: {initial_checkpoint_path}. "
            "Instruction SFT cannot proceed without a valid NTP checkpoint."
        )
    
    if check_checkpoint_compatibility(initial_checkpoint_path, training_module, logging):
        logging.info("✅ NTP checkpoint is compatible, loading model weights...")
        if not load_checkpoint_with_fallback(initial_checkpoint_path, training_module, logging):
            logging.error("❌ Failed to load NTP checkpoint weights!")
            raise RuntimeError(
                f"Could not load NTP checkpoint from {initial_checkpoint_path}. "
                "Instruction SFT requires a valid NTP checkpoint to proceed."
            )
    else:
        logging.warning("⚠️ NTP checkpoint compatibility check failed, attempting to load anyway...")
        if not load_checkpoint_with_fallback(initial_checkpoint_path, training_module, logging):
            logging.error("❌ Failed to load NTP checkpoint weights!")
            raise RuntimeError(
                f"Could not load NTP checkpoint from {initial_checkpoint_path}. "
                "Instruction SFT requires a valid NTP checkpoint to proceed."
            )
    
    logging.info("=" * 80)
    logging.info("✅ NTP checkpoint loaded successfully!")
    logging.info("✅ Model initialized from NTP checkpoint (not from scratch)")
    logging.info("🚀 Ready to start instruction SFT training...")
    logging.info("=" * 80)
    
    # Train with resume capability (for instruction SFT checkpoints)
    # Only include validation dataloader if validation is enabled
    if val_check_interval_steps == float('inf'):
        # Validation disabled - don't pass val_loader
        logging.info("🚀 Starting instruction SFT training without validation...")
        if instruction_sft_checkpoint and instruction_sft_checkpoint != initial_checkpoint_path:
            logging.info(f"🔄 Resuming instruction SFT from checkpoint: {instruction_sft_checkpoint}")
            try:
                trainer.fit(training_module, train_loader, ckpt_path=instruction_sft_checkpoint)
            except Exception as e:
                logging.warning(f"Failed to resume from instruction SFT checkpoint {instruction_sft_checkpoint}: {e}")
                logging.info("Starting instruction SFT training from NTP checkpoint...")
                trainer.fit(training_module, train_loader)
        else:
            logging.info("🚀 Starting instruction SFT training from NTP checkpoint...")
            trainer.fit(training_module, train_loader)
    else:
        # Validation enabled - include val_loader
        logging.info("🚀 Starting instruction SFT training with validation...")
        if instruction_sft_checkpoint and instruction_sft_checkpoint != initial_checkpoint_path:
            logging.info(f"🔄 Resuming instruction SFT from checkpoint: {instruction_sft_checkpoint}")
            try:
                trainer.fit(training_module, train_loader, val_loader, ckpt_path=instruction_sft_checkpoint)
            except Exception as e:
                logging.warning(f"Failed to resume from instruction SFT checkpoint {instruction_sft_checkpoint}: {e}")
                logging.info("Starting instruction SFT training from NTP checkpoint...")
                trainer.fit(training_module, train_loader, val_loader)
        else:
            logging.info("🚀 Starting instruction SFT training from NTP checkpoint...")
            trainer.fit(training_module, train_loader, val_loader)
    
    logging.info("✅ Instruction SFT training completed successfully!")
    return trainer, training_module


def train_megatron_mode_wrapper(
    model_config_key: str = "model_config_1.8B",
    stage: str = "stage1",
    data_path: str = "./data",
    tokenizer_path: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
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
                       choices=["basic", "production"],
                       help="Training mode: basic or production")
    
    # Training backend (for production mode)
    parser.add_argument("--training_backend", type=str, default=None,
                       choices=["lightning", "megatron"],
                       help="Training backend for production mode: lightning (PyTorch Lightning) or megatron (NeMo Megatron). If not specified, reads from config.")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="config_production.yaml",
                       help="Config file path (e.g., config_production.yaml, configs/my_config.yaml, or full path)")
    
    # Common arguments
    parser.add_argument("--stage", type=str, default="stage1_inst_SFT",
                       choices=["stage0", "stage1", "stage1_inst_SFT", "stage2"],
                       help="Training stage (use stage1_inst_SFT for instruction fine-tuning)")
    
    
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (uses config value if not specified)")
    
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    
    parser.add_argument("--devices", type=int, default=None,
                       help="Number of devices (uses config value if not specified)")
    
    parser.add_argument("--precision", type=str, default=None,
                       choices=["16-mixed", "32", "bf16-mixed"],
                       help="Training precision (uses config value if not specified)")
    
    # Resume training arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to instruction SFT checkpoint to resume from (optional)")
    
    parser.add_argument("--ntp_checkpoint_path", type=str, default=None,
                       help="Path to NTP training checkpoint (required for instruction SFT, auto-detected if not provided)")
    
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
    parser.add_argument("--model_config", type=str, default="model_config_1.8B",
                       help="Model configuration key (production mode)")
    
    
    # Dataset processing arguments (only for Lightning backend)
    parser.add_argument("--use_processed_datasets", action="store_true", default=True,
                       help="Use processed datasets for training (Lightning backend only)")
    
    parser.add_argument("--no_processed_datasets", action="store_true", default=False,
                       help="Disable processed datasets and use HuggingFace datasets (Lightning backend only)")
    
    parser.add_argument("--total_samples", type=int, default=None,
                       help="Total number of samples to process for training (overrides config)")
    
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Setup configuration file if specified
    if hasattr(args, 'config') and args.config:
        setup_config_file(args.config)
    
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
            logging.info("🔧 Framework: PyTorch Lightning + Instruction Datasets")
            logging.info("💡 Use case: Instruction fine-tuning after NTP training")
        elif training_backend == "megatron":
            logging.info("📊 Training Method: NeMo Megatron")
            logging.info("🔧 Framework: NeMo Megatron + HuggingFace Datasets")
            logging.info("💡 Use case: Large-scale, distributed training with optimizations")
    
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
                # Use PyTorch Lightning backend for instruction SFT
                # Only pass arguments that are explicitly provided (not None)
                train_kwargs = {
                    "model_config_key": args.model_config,
                    "stage": args.stage,
                    "resume_from_checkpoint": resume_from_checkpoint,
                    "ntp_checkpoint_path": args.ntp_checkpoint_path,
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
                if args.total_samples is not None:
                    train_kwargs["total_samples"] = args.total_samples
                
                trainer, module = train_production_mode(**train_kwargs)
            elif training_backend == "megatron":
                # Use NeMo Megatron backend
                # Only pass arguments that are explicitly provided (not None)
                megatron_kwargs = {
                    "model_config_key": args.model_config,
                    "stage": args.stage,
                    "resume_from_checkpoint": resume_from_checkpoint,
                }
                
                # Only add arguments that are explicitly provided
                if args.data_path is not None:
                    megatron_kwargs["data_path"] = args.data_path
                if args.tokenizer_path is not None:
                    megatron_kwargs["tokenizer_path"] = args.tokenizer_path
                if args.max_length is not None:
                    megatron_kwargs["max_length"] = args.max_length
                if args.learning_rate is not None:
                    megatron_kwargs["learning_rate"] = args.learning_rate
                if args.batch_size is not None:
                    megatron_kwargs["batch_size"] = args.batch_size
                if args.devices is not None:
                    megatron_kwargs["devices"] = args.devices
                if args.precision is not None:
                    megatron_kwargs["precision"] = args.precision
                if args.total_samples is not None:
                    megatron_kwargs["max_samples"] = args.total_samples
                
                trainer, module = train_megatron_mode_wrapper(**megatron_kwargs)
        
        
        
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
                logging.info("📊 Training Method Used: PyTorch Lightning (Instruction SFT)")
            elif training_backend == "megatron":
                logging.info("📊 Training Method Used: NeMo Megatron")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

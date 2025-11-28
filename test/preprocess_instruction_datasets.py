#!/usr/bin/env python3
"""
Instruction Dataset Preprocessing Script

This script downloads and processes instruction tuning datasets for stage1_inst_SFT,
then saves them locally in the data/ directory for fast loading during instruction SFT training.

The script automatically uses configs/config_production.yaml by default (or configs/config.yaml if production doesn't exist).

Usage:
    python test/preprocess_instruction_datasets.py --stage stage1_inst_SFT --total_samples 100000
    python test/preprocess_instruction_datasets.py --stage stage1_inst_SFT --total_samples 100000 --config configs/config_production.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json
import random

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.append(project_root)

from src.utils.tokenizer_manager import get_tokenizer_with_caching

# Import instruction dataset loading function
try:
    from nemo.ModularModelstage1_InstructionSFT import load_instruction_datasets_with_percentages
    from nemo.config_loader import create_nemo_config_from_existing
    print(f"✅ Successfully imported instruction dataset loader from {src_path}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"💡 Looking for module in: {src_path}")
    print("💡 Make sure you're running this in the correct environment with all dependencies installed")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_instruction_datasets(
    stage: str = "stage1_inst_SFT", 
    total_samples: int = 100000, 
    config_path: str = "configs/config_production.yaml",
    model_config_key: str = "model_config_1.8B"
):
    """
    Preprocess instruction datasets for the specified stage and save them locally.
    
    Args:
        stage: Training stage ("stage1_inst_SFT")
        total_samples: Total number of samples to process
        config_path: Path to the config.yaml file
        model_config_key: Model configuration key
    """
    
    logger.info(f"🚀 Starting instruction dataset preprocessing for {stage}")
    logger.info(f"📊 Target samples: {total_samples:,}")
    logger.info(f"⚙️  Config: {config_path}")
    logger.info(f"🤖 Model config: {model_config_key}")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    stage_dir = data_dir / stage
    stage_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    raw_dir = stage_dir / "raw"
    processed_dir = stage_dir / "processed"
    metadata_dir = stage_dir / "metadata"
    
    for dir_path in [raw_dir, processed_dir, metadata_dir]:
        dir_path.mkdir(exist_ok=True)
    
    logger.info(f"📁 Data directory: {stage_dir.absolute()}")
    
    try:
        # Load configuration
        logger.info("🔧 Loading configuration...")
        config = create_nemo_config_from_existing(model_config_key, stage)
        
        # Get instruction datasets configuration
        data_config = config.get("data", {})
        processing_config = data_config.get("processing", {})
        pretraining_datasets = data_config.get("pretraining_datasets", {})
        
        if not pretraining_datasets:
            logger.error("❌ No instruction datasets found in config!")
            logger.error("💡 Please configure pretraining_datasets in the config file for stage1_inst_SFT")
            return False
        
        # Get validation split ratio from config (default 0.1 = 10%)
        validation_split_ratio = processing_config.get("validation_split_ratio", 0.1)
        
        seed = processing_config.get("seed", 42)
        rng = random.Random(seed)
        
        logger.info(f"📚 Found {len(pretraining_datasets)} instruction datasets in config")
        logger.info(f"📊 Validation split ratio: {validation_split_ratio:.1%}")
        
        # ------------------------------------------------------------------
        # Load tokenizer with instruction-format special tokens (same as training)
        # ------------------------------------------------------------------
        tokenizer_path = (
            config.get("tokenizer_path")
            or config.get("model", {}).get("tokenizer_path")
            or data_config.get("tokenizer_path")
            or "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        )
        logger.info(f"🔤 Loading tokenizer for preprocessing: {tokenizer_path}")
        
        tokenizer = get_tokenizer_with_caching(
            tokenizer_path=tokenizer_path,
            custom_tokens=None,  # Adds default custom tokens via TokenizerManager
            force_download=False,
            cache_dir="tokenizers"
        )
        
        instruction_tokens = ["<|instruction|>", "<|response|>", "<|end|>"]
        missing_tokens = [tok for tok in instruction_tokens if tok not in tokenizer.get_vocab()]
        
        if missing_tokens:
            logger.info(f"➕ Adding instruction tokens: {missing_tokens}")
            tokenizer.add_tokens(missing_tokens)
            logger.info(f"✅ Tokenizer vocab size after additions: {len(tokenizer)}")
        else:
            logger.info("✅ All instruction-format tokens already present in tokenizer")
        
        # Ensure pad token is distinct from eos (parity with training)
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                logger.info(f"🔧 pad_token_id set to unk_token_id ({tokenizer.unk_token_id})")
            else:
                tokenizer.pad_token_id = 0
                logger.info("🔧 pad_token_id not set; defaulting to 0")
        
        # Categorize datasets by allocation strategy
        priority_datasets = []
        percentage_datasets = []
        for dataset_name, dataset_config in pretraining_datasets.items():
            if dataset_config.get("use_all_available", False):
                priority_datasets.append(dataset_name)
                logger.info(f"  - {dataset_name}: use_all_available (priority dataset)")
            elif dataset_config.get("percentage", 0) > 0:
                percentage_datasets.append(dataset_name)
                logger.info(f"  - {dataset_name}: {dataset_config.get('percentage')}% (scales with remaining samples)")
            else:
                logger.warning(f"  - {dataset_name}: No allocation strategy specified!")
        
        logger.info(f"📊 Allocation strategy: {len(priority_datasets)} priority datasets, {len(percentage_datasets)} percentage-based datasets")
        
        # Calculate validation samples based on configurable ratio
        val_samples = max(int(total_samples * validation_split_ratio), 100)  # At least 100 validation samples
        if val_samples >= total_samples:
            logger.warning(
                f"⚠️ validation_split_ratio ({validation_split_ratio:.2%}) results in "
                f"{val_samples:,} validation samples >= total_samples ({total_samples:,}). "
                "Reducing validation samples to keep at least 1 training sample."
            )
            val_samples = max(total_samples - 1, 0)
        train_samples = max(total_samples - val_samples, 0)
        logger.info(
            f"📊 Sample allocation -> train: {train_samples:,} ({(train_samples / total_samples * 100) if total_samples else 0:.2f}%), "
            f"val: {val_samples:,} ({(val_samples / total_samples * 100) if total_samples else 0:.2f}%)"
        )
        
        def partition_samples(samples, target_count, split_name):
            if target_count <= 0:
                logger.warning(f"⚠️ Target {split_name} samples set to <= 0. Returning empty list.")
                return [], samples
            current_count = len(samples)
            if current_count <= target_count:
                if current_count < target_count:
                    logger.warning(
                        f"⚠️ Requested {target_count:,} {split_name} samples but only {current_count:,} available."
                    )
                return samples, []
            rng.shuffle(samples)
            logger.info(
                f"✂️ Partitioning {split_name} dataset from {current_count:,} to {target_count:,} samples "
                f"to respect validation_split_ratio."
            )
            return samples[:target_count], samples[target_count:]
        
        # Load instruction datasets
        logger.info("📥 Loading instruction datasets...")
        start_time = time.time()
        
        train_request_samples = max(total_samples, train_samples)
        train_instruction_data = load_instruction_datasets_with_percentages(
            pretraining_datasets, 
            train_request_samples, 
            "train"
        )
        
        val_instruction_data = load_instruction_datasets_with_percentages(
            pretraining_datasets, 
            val_samples,
            "validation"
        )
        
        # Allocate validation target and recycle any excess into the training pool
        val_instruction_data, val_excess = partition_samples(val_instruction_data, val_samples, "validation")
        if val_excess:
            train_instruction_data.extend(val_excess)
        
        # Finally, cap training samples to requested count
        train_instruction_data, train_excess = partition_samples(train_instruction_data, train_samples, "train")
        if train_excess:
            logger.info(
                f"♻️  Discarding {len(train_excess):,} surplus training samples after enforcing split targets."
            )
        
        load_time = time.time() - start_time
        actual_train_samples = len(train_instruction_data)
        actual_val_samples = len(val_instruction_data)
        logger.info(f"✅ Loaded {actual_train_samples:,} training and {actual_val_samples:,} validation samples in {load_time:.2f} seconds")
        
        # Save the datasets
        logger.info("💾 Saving instruction datasets...")
        save_start = time.time()
        
        # Convert to HuggingFace dataset format for saving
        from datasets import Dataset as HFDataset
        
        train_dataset = HFDataset.from_list(train_instruction_data)
        val_dataset = HFDataset.from_list(val_instruction_data)
        
        # Save as HuggingFace dataset format
        train_dataset.save_to_disk(str(processed_dir / "train_dataset"))
        val_dataset.save_to_disk(str(processed_dir / "val_dataset"))
        
        # Also save combined for convenience
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets([train_dataset, val_dataset])
        combined_dataset.save_to_disk(str(processed_dir / "combined_dataset"))
        
        # Save as JSON for easy inspection
        sample_data = {
            "train_samples": train_instruction_data[:50],  # First 50 training samples
            "val_samples": val_instruction_data[:10]      # First 10 validation samples
        }
        
        with open(processed_dir / "sample_data.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        save_time = time.time() - save_start
        logger.info(f"✅ Saved datasets in {save_time:.2f} seconds")
        
        # Save metadata
        metadata = {
            "stage": stage,
            "total_samples": total_samples,
            "train_samples": len(train_instruction_data),
            "val_samples": len(val_instruction_data),
            "preprocessing_time": load_time,
            "save_time": save_time,
            "config_path": config_path,
            "model_config_key": model_config_key,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets_config": pretraining_datasets,
            "dataset_format": "instruction_format",
            "validation_split_ratio": validation_split_ratio,
            "tokenizer_path": tokenizer_path,
            "tokenizer_vocab_size": len(tokenizer),
            "instruction_tokens": instruction_tokens
        }
        
        with open(metadata_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual dataset information
        # Calculate expected samples based on allocation strategy (similar to preprocess_datasets.py)
        individual_datasets = {}
        fixed_samples_total = 0
        
        # First pass: calculate fixed samples (use_all_available)
        for dataset_name, config_data in pretraining_datasets.items():
            if config_data.get("use_all_available", False):
                # Will use all available - we don't know exact count, but note it
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "use_all_available",
                    "expected_train_samples": "all_available",  # Unknown until processed
                    "expected_val_samples": "all_available",  # Unknown until processed
                    "subset": config_data.get("subset"),
                    "task_type": "instruction"
                }
            elif config_data.get("percentage", 0) > 0:
                # Will be calculated in second pass based on remaining samples
                pass
        
        # Estimate fixed samples for priority datasets (rough estimate for metadata)
        # Note: Actual counts will be determined when datasets are loaded
        priority_estimate = 4_486_000  # From config comments
        fixed_samples_total = priority_estimate
        
        # Second pass: calculate percentage-based allocations
        remaining_samples = total_samples - fixed_samples_total
        if remaining_samples < 0:
            remaining_samples = 0
            logger.warning(f"⚠️  Priority datasets ({fixed_samples_total:,}) may exceed total_samples ({total_samples:,})")
        
        percentage_datasets_dict = {}
        for dataset_name, config_data in pretraining_datasets.items():
            if dataset_name not in individual_datasets:  # Not already processed
                percentage = config_data.get("percentage", 0)
                if percentage > 0:
                    percentage_datasets_dict[dataset_name] = percentage
        
        # Normalize percentages and calculate expected samples
        total_percentage = sum(percentage_datasets_dict.values())
        if total_percentage > 0 and remaining_samples > 0:
            for dataset_name, percentage in percentage_datasets_dict.items():
                config_data = pretraining_datasets[dataset_name]
                # Calculate based on normalized percentage of remaining samples
                expected_train_samples = int(remaining_samples * (percentage / total_percentage))
                expected_val_samples = max(int(expected_train_samples * validation_split_ratio), 10)
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "percentage",
                    "percentage": percentage,
                    "expected_train_samples": expected_train_samples,
                    "expected_val_samples": expected_val_samples,
                    "subset": config_data.get("subset"),
                    "task_type": "instruction"
                }
        elif total_percentage > 0:
            # No remaining samples after fixed allocations
            for dataset_name, percentage in percentage_datasets_dict.items():
                config_data = pretraining_datasets[dataset_name]
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "percentage",
                    "percentage": percentage,
                    "expected_train_samples": 0,  # No samples available
                    "expected_val_samples": 0,
                    "subset": config_data.get("subset"),
                    "task_type": "instruction"
                }
        
        with open(metadata_dir / "individual_datasets.json", 'w') as f:
            json.dump(individual_datasets, f, indent=2)
        
        logger.info("📋 Saved metadata files")
        
        # Print summary
        actual_total_samples = actual_train_samples + actual_val_samples
        samples_difference = actual_total_samples - total_samples
        
        logger.info("=" * 60)
        logger.info("📊 PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Stage: {stage}")
        logger.info(f"Target samples: {total_samples:,}")
        logger.info(f"Training samples: {actual_train_samples:,}")
        logger.info(f"Validation samples: {actual_val_samples:,}")
        logger.info(f"Total samples: {actual_total_samples:,}")
        if samples_difference != 0:
            diff_percent = (samples_difference / total_samples) * 100 if total_samples > 0 else 0
            logger.info(f"Difference: {samples_difference:+,} ({diff_percent:+.2f}%)")
            if abs(diff_percent) > 1.0:  # More than 1% difference
                logger.warning(f"⚠️  Sample count differs from target by {abs(diff_percent):.2f}%")
        else:
            logger.info("✅ Sample count matches target exactly!")
        logger.info(f"Loading time: {load_time:.2f} seconds")
        logger.info(f"Saving time: {save_time:.2f} seconds")
        logger.info(f"Total time: {load_time + save_time:.2f} seconds")
        logger.info(f"Data directory: {stage_dir.absolute()}")
        logger.info("=" * 60)
        
        # Print individual dataset breakdown
        logger.info("📈 DATASET BREAKDOWN:")
        for dataset_name, info in individual_datasets.items():
            strategy = info.get("allocation_strategy", "unknown")
            if strategy == "use_all_available":
                logger.info(f"  - {dataset_name}: all available samples ({strategy})")
            elif strategy == "percentage":
                pct = info.get("percentage", 0)
                train_samples = info.get("expected_train_samples", 0)
                val_samples = info.get("expected_val_samples", 0)
                if isinstance(train_samples, int) and isinstance(val_samples, int):
                    logger.info(f"  - {dataset_name}: {train_samples:,} train + {val_samples:,} val ({pct}% of remaining)")
                else:
                    logger.info(f"  - {dataset_name}: {train_samples} train + {val_samples} val ({pct}% of remaining)")
            else:
                train_samples = info.get("expected_train_samples", "unknown")
                val_samples = info.get("expected_val_samples", "unknown")
                logger.info(f"  - {dataset_name}: {train_samples} train + {val_samples} val")
        
        logger.info("=" * 60)
        logger.info("✅ Instruction dataset preprocessing completed successfully!")
        logger.info(f"💡 Datasets saved to: {processed_dir.absolute()}")
        logger.info(f"💡 Training script will automatically use these preprocessed datasets")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Instruction dataset preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_preprocessed_instruction_dataset(stage: str = "stage1_inst_SFT"):
    """
    Load a preprocessed instruction dataset from the data directory.
    
    Args:
        stage: Training stage to load
        
    Returns:
        Tuple of (train_dataset, val_dataset) or None if not found
    """
    from datasets import load_from_disk
    
    processed_dir = Path("data") / stage / "processed"
    train_path = processed_dir / "train_dataset"
    val_path = processed_dir / "val_dataset"
    
    if not train_path.exists() or not val_path.exists():
        logger.error(f"❌ Preprocessed instruction datasets not found at {processed_dir}")
        logger.info(f"💡 Looking for: {train_path} and {val_path}")
        return None
    
    logger.info(f"📥 Loading preprocessed instruction datasets from {processed_dir}")
    train_dataset = load_from_disk(str(train_path))
    val_dataset = load_from_disk(str(val_path))
    
    logger.info(f"✅ Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Preprocess instruction datasets for NeMo instruction SFT training")
    
    parser.add_argument("--stage", type=str, default="stage1_inst_SFT", 
                       choices=["stage1_inst_SFT"],
                       help="Training stage to preprocess (currently only stage1_inst_SFT supported)")
    
    parser.add_argument("--total_samples", type=int, default=10000000,
                       help="Total number of samples to process")
    
    parser.add_argument("--config", type=str, default="configs/config_production.yaml",
                       help="Path to config.yaml file")
    
    parser.add_argument("--model_config", type=str, default="model_config_1.8B",
                       help="Model configuration key")
    
    parser.add_argument("--load_only", action="store_true",
                       help="Only load and display preprocessed dataset info")
    
    args = parser.parse_args()
    
    if args.load_only:
        # Just load and display info about preprocessed dataset
        result = load_preprocessed_instruction_dataset(args.stage)
        if result:
            train_dataset, val_dataset = result
            logger.info(f"✅ Successfully loaded preprocessed instruction datasets")
            logger.info(f"   Training samples: {len(train_dataset):,}")
            logger.info(f"   Validation samples: {len(val_dataset):,}")
            
            # Show sample structure
            if len(train_dataset) > 0:
                logger.info(f"\n📋 Sample structure:")
                sample = train_dataset[0]
                logger.info(f"   Keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, str):
                        logger.info(f"   {key}: {value[:100]}..." if len(value) > 100 else f"   {key}: {value}")
                    else:
                        logger.info(f"   {key}: {type(value).__name__}")
        else:
            logger.error("❌ Failed to load preprocessed instruction datasets")
            sys.exit(1)
    else:
        # Preprocess datasets
        success = preprocess_instruction_datasets(
            stage=args.stage,
            total_samples=args.total_samples,
            config_path=args.config,
            model_config_key=args.model_config
        )
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Instruction Dataset Preprocessing Script

This script downloads and processes instruction tuning datasets for stage1_inst_SFT,
then saves them locally in the data/ directory for fast loading during instruction SFT training.

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

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.append(project_root)

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
        pretraining_datasets = data_config.get("pretraining_datasets", {})
        
        if not pretraining_datasets:
            logger.error("❌ No instruction datasets found in config!")
            logger.error("💡 Please configure pretraining_datasets in the config file for stage1_inst_SFT")
            return False
        
        logger.info(f"📚 Found {len(pretraining_datasets)} instruction datasets in config")
        for dataset_name, dataset_config in pretraining_datasets.items():
            percentage = dataset_config.get("percentage", 0)
            logger.info(f"  - {dataset_name}: {percentage}%")
        
        # Load instruction datasets
        logger.info("📥 Loading instruction datasets...")
        start_time = time.time()
        
        train_instruction_data = load_instruction_datasets_with_percentages(
            pretraining_datasets, 
            total_samples, 
            "train"
        )
        
        val_instruction_data = load_instruction_datasets_with_percentages(
            pretraining_datasets, 
            max(total_samples // 10, 100),  # At least 100 validation samples
            "validation"
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Loaded {len(train_instruction_data)} training and {len(val_instruction_data)} validation samples in {load_time:.2f} seconds")
        
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
            "dataset_format": "instruction_format"
        }
        
        with open(metadata_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual dataset information
        individual_datasets = {}
        for dataset_name, config_data in pretraining_datasets.items():
            percentage = config_data.get("percentage", 0)
            expected_samples = int(total_samples * (percentage / 100.0))
            individual_datasets[dataset_name] = {
                "percentage": percentage,
                "expected_train_samples": expected_samples,
                "expected_val_samples": max(int(expected_samples // 10), 10),
                "subset": config_data.get("subset"),
            }
        
        with open(metadata_dir / "individual_datasets.json", 'w') as f:
            json.dump(individual_datasets, f, indent=2)
        
        logger.info("📋 Saved metadata files")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Stage: {stage}")
        logger.info(f"Training samples: {len(train_instruction_data):,}")
        logger.info(f"Validation samples: {len(val_instruction_data):,}")
        logger.info(f"Total samples: {len(combined_dataset):,}")
        logger.info(f"Loading time: {load_time:.2f} seconds")
        logger.info(f"Saving time: {save_time:.2f} seconds")
        logger.info(f"Total time: {load_time + save_time:.2f} seconds")
        logger.info(f"Data directory: {stage_dir.absolute()}")
        logger.info("=" * 60)
        
        # Print individual dataset breakdown
        logger.info("📈 DATASET BREAKDOWN:")
        for dataset_name, info in individual_datasets.items():
            logger.info(f"  - {dataset_name}: {info['expected_train_samples']:,} train + {info['expected_val_samples']:,} val ({info['percentage']}%)")
        
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
    
    parser.add_argument("--total_samples", type=int, default=100000,
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


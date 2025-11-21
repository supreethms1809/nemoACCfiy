#!/usr/bin/env python3
"""
Dataset Preprocessing Script

This script downloads and processes datasets based on the config.yaml configuration,
then saves them locally in the data/ directory for fast loading during training.

Usage:
    python test/preprocess_datasets.py --stage stage1 --total_samples 1000000
    python test/preprocess_datasets.py --stage stage1 --config configs/config_production.yaml --total_samples 1000000
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

# Import only what we need to avoid torch dependency issues
try:
    from nemo.huggingface_dataset_loader import HuggingFaceDatasetLoader
    print(f"✅ Successfully imported HuggingFaceDatasetLoader from {src_path}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"💡 Looking for module in: {src_path}")
    print("💡 Make sure you're running this in the correct environment with all dependencies installed")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_datasets(stage: str = "stage1", total_samples: int = 1000000, config_path: str = "configs/config_production.yaml"):
    """
    Preprocess datasets for the specified stage and save them locally.
    
    Args:
        stage: Training stage ("stage1", "stage2", "stage3")
        total_samples: Total number of samples to process
        config_path: Path to the config.yaml file
    """
    
    logger.info(f"🚀 Starting dataset preprocessing for {stage}")
    logger.info(f"📊 Target samples: {total_samples:,}")
    logger.info(f"⚙️  Config: {config_path}")
    
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
        # Initialize the dataset loader (disable buffer for preprocessing efficiency)
        logger.info("🔧 Initializing HuggingFace dataset loader...")
        hf_loader = HuggingFaceDatasetLoader(config_path, stage=stage, enable_buffer=False)
        
        # Load all datasets using streaming approach
        logger.info("📥 Loading datasets using streaming approach...")
        start_time = time.time()
        
        # Use the new streaming create_training_data method
        training_samples = hf_loader.create_training_data(total_samples=total_samples)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Loaded {len(training_samples)} samples in {load_time:.2f} seconds")
        
        # Save the combined dataset
        logger.info("💾 Saving combined dataset...")
        save_start = time.time()
        
        # Convert to HuggingFace dataset format for saving
        from datasets import Dataset as HFDataset
        combined_dataset = HFDataset.from_list(training_samples)
        
        # Save as HuggingFace dataset format
        combined_dataset.save_to_disk(str(processed_dir / "combined_dataset"))
        
        # Save as JSON for easy inspection
        sample_data = training_samples[:100]  # Save first 100 samples as examples
        
        with open(processed_dir / "sample_data.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        save_time = time.time() - save_start
        logger.info(f"✅ Saved dataset in {save_time:.2f} seconds")
        
        # Save metadata
        metadata = {
            "stage": stage,
            "total_samples": total_samples,
            "actual_samples": len(training_samples),
            "preprocessing_time": load_time,
            "save_time": save_time,
            "config_path": config_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets_config": hf_loader.datasets_config,
            "stage_processing": hf_loader.stage_processing,
            "global_settings": hf_loader.global_settings,
            "streaming_approach": True
        }
        
        with open(metadata_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual dataset information
        individual_datasets = {}
        for dataset_name, config in hf_loader.datasets_config.items():
            percentage = config.get("percentage", 0)
            expected_samples = int(total_samples * (percentage / 100.0))
            individual_datasets[dataset_name] = {
                "percentage": percentage,
                "expected_samples": expected_samples,
                "subset": config.get("subset"),
                "task_type": config.get("task_type", "general")
            }
        
        with open(metadata_dir / "individual_datasets.json", 'w') as f:
            json.dump(individual_datasets, f, indent=2)
        
        logger.info("📋 Saved metadata files")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Stage: {stage}")
        logger.info(f"Total samples processed: {len(combined_dataset):,}")
        logger.info(f"Loading time: {load_time:.2f} seconds")
        logger.info(f"Saving time: {save_time:.2f} seconds")
        logger.info(f"Total time: {load_time + save_time:.2f} seconds")
        logger.info(f"Data directory: {stage_dir.absolute()}")
        logger.info("=" * 60)
        
        # Print individual dataset breakdown
        logger.info("📈 DATASET BREAKDOWN:")
        for dataset_name, info in individual_datasets.items():
            logger.info(f"  - {dataset_name}: {info['expected_samples']:,} samples ({info['percentage']}%)")
        
        logger.info("=" * 60)
        logger.info("✅ Dataset preprocessing completed successfully!")
        logger.info(f"💡 To use preprocessed data, modify your config to set 'use_processed_datasets: true'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_preprocessed_dataset(stage: str = "stage1"):
    """
    Load a preprocessed dataset from the data directory.
    
    Args:
        stage: Training stage to load
        
    Returns:
        Loaded dataset or None if not found
    """
    from datasets import load_from_disk
    
    processed_dir = Path("data") / stage / "processed"
    dataset_path = processed_dir / "combined_dataset"
    
    if not dataset_path.exists():
        logger.error(f"❌ Preprocessed dataset not found at {dataset_path}")
        return None
    
    logger.info(f"📥 Loading preprocessed dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    logger.info(f"✅ Loaded {len(dataset)} samples")
    
    return dataset

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for NeMo training")
    
    parser.add_argument("--stage", type=str, default="stage1", 
                       choices=["stage1", "stage2", "stage3"],
                       help="Training stage to preprocess")
    
    parser.add_argument("--total_samples", type=int, default=10000000,
                       help="Total number of samples to process")
    
    parser.add_argument("--config", type=str, default="configs/config_production.yaml",
                       help="Path to config.yaml file (default: configs/config_production.yaml)")
    
    parser.add_argument("--load_only", action="store_true",
                       help="Only load and display preprocessed dataset info")
    
    args = parser.parse_args()
    
    if args.load_only:
        # Just load and display info about preprocessed dataset
        dataset = load_preprocessed_dataset(args.stage)
        if dataset:
            logger.info(f"✅ Successfully loaded preprocessed dataset with {len(dataset)} samples")
        else:
            logger.error("❌ Failed to load preprocessed dataset")
            sys.exit(1)
    else:
        # Preprocess datasets
        success = preprocess_datasets(
            stage=args.stage,
            total_samples=args.total_samples,
            config_path=args.config
        )
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()

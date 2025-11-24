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
import shutil
import gc

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
        actual_samples_count = len(training_samples)  # Save count before any deletions
        logger.info(f"✅ Loaded {actual_samples_count:,} samples in {load_time:.2f} seconds")
        
        # Check disk space before saving (rough estimate: assume ~1KB per sample, plus overhead)
        estimated_size_gb = (actual_samples_count * 1024) / (1024**3) * 2  # 2x safety margin
        try:
            stat = shutil.disk_usage(processed_dir)
            available_gb = stat.free / (1024**3)
            logger.info(f"💾 Disk space check: {available_gb:.2f} GB available, ~{estimated_size_gb:.2f} GB estimated needed")
            if available_gb < estimated_size_gb:
                logger.warning(f"⚠️  Warning: Low disk space! Available: {available_gb:.2f} GB, Estimated need: {estimated_size_gb:.2f} GB")
        except Exception as e:
            logger.warning(f"⚠️  Could not check disk space: {e}")
        
        # Check memory usage
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / (1024**3)
            logger.info(f"🧠 Current memory usage: {mem_gb:.2f} GB")
        except ImportError:
            logger.info("💡 Install 'psutil' for memory monitoring: pip install psutil")
        except Exception as e:
            logger.warning(f"⚠️  Could not check memory: {e}")
        
        # Save sample data JSON first (before any memory cleanup)
        logger.info("💾 Saving sample data JSON...")
        sample_data = training_samples[:100]  # Save first 100 samples as examples
        with open(processed_dir / "sample_data.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        del sample_data  # Free up sample data
        
        # Save the combined dataset
        logger.info("💾 Saving combined dataset...")
        save_start = time.time()
        dataset_path = str(processed_dir / "combined_dataset")
        
        try:
            # For very large datasets, use chunked conversion to avoid OOM
            num_samples = len(training_samples)
            chunk_size = 5_000_000  # Process 5M samples at a time
            
            if num_samples > chunk_size:
                logger.info(f"📦 Converting to HuggingFace Dataset format in chunks (chunk size: {chunk_size:,})...")
                logger.info(f"💡 This will process {num_samples:,} samples in {(num_samples + chunk_size - 1) // chunk_size} chunks")
                
                from datasets import Dataset as HFDataset, concatenate_datasets
                from datasets import load_from_disk
                
                # Process in chunks and save temporarily
                chunk_datasets = []
                temp_chunk_dir = processed_dir / "temp_chunks"
                temp_chunk_dir.mkdir(exist_ok=True)
                
                try:
                    for chunk_idx in range(0, num_samples, chunk_size):
                        chunk_end = min(chunk_idx + chunk_size, num_samples)
                        chunk_data = training_samples[chunk_idx:chunk_end]
                        
                        logger.info(f"📦 Processing chunk {chunk_idx // chunk_size + 1}/{(num_samples + chunk_size - 1) // chunk_size} "
                                  f"(samples {chunk_idx:,} to {chunk_end:,})...")
                        
                        # Convert chunk to dataset
                        chunk_dataset = HFDataset.from_list(chunk_data)
                        del chunk_data  # Free chunk data immediately
                        gc.collect()
                        
                        # Save chunk to temp directory
                        chunk_path = str(temp_chunk_dir / f"chunk_{chunk_idx // chunk_size}")
                        chunk_dataset.save_to_disk(chunk_path)
                        chunk_datasets.append(load_from_disk(chunk_path))
                        
                        # Clean up
                        del chunk_dataset
                        gc.collect()
                        
                        # Log memory
                        try:
                            import psutil
                            process = psutil.Process()
                            mem_info = process.memory_info()
                            mem_gb = mem_info.rss / (1024**3)
                            logger.info(f"   🧠 Memory after chunk {chunk_idx // chunk_size + 1}: {mem_gb:.2f} GB")
                        except:
                            pass
                    
                    # Now that training_samples is fully processed, we can free it
                    del training_samples
                    gc.collect()
                    logger.info("🧹 Freed training_samples from memory")
                    
                    # Concatenate chunks
                    logger.info("📦 Concatenating chunks into final dataset...")
                    combined_dataset = concatenate_datasets(chunk_datasets)
                    logger.info(f"✅ Created combined dataset with {len(combined_dataset):,} samples")
                    
                    # Free chunk datasets
                    del chunk_datasets
                    gc.collect()
                    
                    # Save final dataset
                    logger.info(f"💾 Saving final dataset to {dataset_path}...")
                    combined_dataset.save_to_disk(dataset_path)
                    logger.info(f"✅ Dataset saved successfully!")
                    
                    # Clean up temp chunks
                    logger.info("🧹 Cleaning up temporary chunk files...")
                    import shutil
                    shutil.rmtree(temp_chunk_dir)
                    
                except Exception as e:
                    # Clean up temp chunks on error
                    if temp_chunk_dir.exists():
                        try:
                            import shutil
                            shutil.rmtree(temp_chunk_dir)
                        except:
                            pass
                    raise
            else:
                # Small enough to convert directly
                logger.info("📦 Converting to HuggingFace Dataset format...")
                from datasets import Dataset as HFDataset
                combined_dataset = HFDataset.from_list(training_samples)
                logger.info(f"✅ Converted {len(combined_dataset)} samples to HF Dataset format")
                
                # Free up training_samples list to save memory before disk write
                del training_samples
                gc.collect()
                logger.info("🧹 Freed training_samples from memory")
                
                # Check memory after conversion
                try:
                    import psutil
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    mem_gb = mem_info.rss / (1024**3)
                    logger.info(f"🧠 Memory usage after conversion: {mem_gb:.2f} GB")
                except:
                    pass
                
                # Save as HuggingFace dataset format
                logger.info(f"💾 Saving dataset to {dataset_path}...")
                logger.info("⏳ This may take a while for large datasets...")
                combined_dataset.save_to_disk(dataset_path)
                logger.info(f"✅ Dataset saved successfully!")
            
            # Verify the save worked
            if Path(dataset_path).exists():
                # Check actual size
                try:
                    total_size = sum(f.stat().st_size for f in Path(dataset_path).rglob('*') if f.is_file())
                    size_gb = total_size / (1024**3)
                    logger.info(f"📊 Dataset size: {size_gb:.2f} GB")
                except:
                    pass
                logger.info("✅ Dataset directory created and verified")
            else:
                raise RuntimeError(f"Dataset save failed: {dataset_path} does not exist")
            
            # Clean up combined_dataset
            del combined_dataset
            gc.collect()
            logger.info("🧹 Freed combined_dataset from memory")
            
        except MemoryError as e:
            logger.error(f"❌ Out of memory while saving dataset: {e}")
            logger.error("💡 Try reducing total_samples or run on a machine with more RAM")
            logger.error("💡 The dataset may be too large to save in memory. Consider saving in chunks.")
            raise
        except OSError as e:
            if "No space left on device" in str(e) or "28" in str(e):  # 28 is ENOSPC error code
                logger.error(f"❌ Out of disk space while saving dataset: {e}")
                logger.error("💡 Free up disk space and try again")
                # Try to check disk space again
                try:
                    stat = shutil.disk_usage(processed_dir)
                    available_gb = stat.free / (1024**3)
                    logger.error(f"💾 Available disk space: {available_gb:.2f} GB")
                except:
                    pass
            raise
        except KeyboardInterrupt:
            logger.warning("⚠️  Save interrupted by user")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to save dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        save_time = time.time() - save_start
        logger.info(f"✅ Saved dataset in {save_time:.2f} seconds")
        
        # Save metadata
        metadata = {
            "stage": stage,
            "total_samples": total_samples,
            "actual_samples": actual_samples_count,
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
        # Calculate expected samples based on allocation strategy
        individual_datasets = {}
        fixed_samples_total = 0
        
        # First pass: calculate fixed samples (use_all_available and max_samples)
        for dataset_name, config in hf_loader.datasets_config.items():
            if config.get("use_all_available", False):
                # Will use all available - we don't know exact count, but note it
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "use_all_available",
                    "expected_samples": "all_available",  # Unknown until processed
                    "subset": config.get("subset"),
                    "task_type": config.get("task_type", "general")
                }
            elif "max_samples" in config:
                max_samples = config.get("max_samples")
                fixed_samples_total += max_samples
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "max_samples",
                    "max_samples": max_samples,
                    "expected_samples": max_samples,
                    "subset": config.get("subset"),
                    "task_type": config.get("task_type", "general")
                }
        
        # Second pass: calculate percentage-based allocations
        remaining_samples = total_samples - fixed_samples_total
        percentage_datasets = {}
        for dataset_name, config in hf_loader.datasets_config.items():
            if dataset_name not in individual_datasets:  # Not already processed
                percentage = config.get("percentage", 0)
                if percentage > 0:
                    percentage_datasets[dataset_name] = percentage
        
        # Normalize percentages and calculate expected samples
        total_percentage = sum(percentage_datasets.values())
        if total_percentage > 0 and remaining_samples > 0:
            for dataset_name, percentage in percentage_datasets.items():
                config = hf_loader.datasets_config[dataset_name]
                # Calculate based on normalized percentage of remaining samples
                expected_samples = int(remaining_samples * (percentage / total_percentage))
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "percentage",
                    "percentage": percentage,
                    "expected_samples": expected_samples,
                    "subset": config.get("subset"),
                    "task_type": config.get("task_type", "general")
                }
        elif total_percentage > 0:
            # No remaining samples after fixed allocations
            for dataset_name, percentage in percentage_datasets.items():
                config = hf_loader.datasets_config[dataset_name]
                individual_datasets[dataset_name] = {
                    "allocation_strategy": "percentage",
                    "percentage": percentage,
                    "expected_samples": 0,  # No samples available
                    "subset": config.get("subset"),
                    "task_type": config.get("task_type", "general")
                }
        
        with open(metadata_dir / "individual_datasets.json", 'w') as f:
            json.dump(individual_datasets, f, indent=2)
        
        logger.info("📋 Saved metadata files")
        
        # Print summary
        actual_samples = actual_samples_count
        samples_difference = actual_samples - total_samples
        
        logger.info("=" * 60)
        logger.info("📊 PREPROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Stage: {stage}")
        logger.info(f"Target samples: {total_samples:,}")
        logger.info(f"Actual samples processed: {actual_samples:,}")
        if samples_difference != 0:
            diff_percent = (samples_difference / total_samples) * 100
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
            elif strategy == "max_samples":
                logger.info(f"  - {dataset_name}: {info['expected_samples']:,} samples ({strategy})")
            elif strategy == "percentage":
                pct = info.get("percentage", 0)
                samples = info.get("expected_samples", 0)
                logger.info(f"  - {dataset_name}: {samples:,} samples ({pct}% of remaining)")
            else:
                logger.info(f"  - {dataset_name}: {info.get('expected_samples', 'unknown')} samples")
        
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
    
    parser.add_argument("--total_samples", type=int, default=20000000,
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

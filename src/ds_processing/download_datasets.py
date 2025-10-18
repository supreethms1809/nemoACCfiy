#!/usr/bin/env python3
"""
Dataset Downloader for NeMo ModularModel Pretraining

This script downloads and caches Hugging Face pretraining datasets
for offline processing and training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available. Install with: pip install datasets")

class DatasetDownloader:
    """Downloads and caches Hugging Face datasets."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the downloader."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Dataset list from the Hugging Face collection
        self.datasets = {
            # Web datasets
            "HuggingFaceFW/fineweb": "High quality web content (52.5B tokens)",
            "HuggingFaceFW/fineweb-2": "Additional web content (5.02B tokens)", 
            "HuggingFaceFW/fineweb-edu": "Educational web content (3.5B tokens)",
            
            # Code datasets
            "bigcode/the-stack": "GitHub code (546M tokens)",
            "bigcode/the-stack-v2": "Updated GitHub code (5.45B tokens)",
            
            # Math and reasoning
            "HuggingFaceTB/finemath": "Math content (48.3M tokens)",
            "mlfoundations/dclm-baseline-1.0": "Curated content (187k tokens)",
            
            # Synthetic educational content
            "HuggingFaceTB/cosmopedia": "Synthetic textbooks (31.1M tokens)",
            "HuggingFaceTB/smollm-corpus": "Educational content (237M tokens)",
        }
    
    def list_datasets(self):
        """List available datasets."""
        print("📚 Available Pretraining Datasets:")
        print("=" * 60)
        for i, (name, description) in enumerate(self.datasets.items(), 1):
            print(f"{i:2d}. {name}")
            print(f"    {description}")
            print()
    
    def download_dataset(self, dataset_name: str, max_samples: int = None):
        """Download a single dataset."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets not available")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.logger.info(f"📥 Downloading dataset: {dataset_name}")
        
        try:
            # Try streaming first for large datasets
            try:
                dataset = load_dataset(
                    dataset_name,
                    streaming=True,
                    cache_dir=str(self.cache_dir)
                )
                self.logger.info(f"✅ Downloaded {dataset_name} (streaming mode)")
                return dataset
            except Exception as e:
                self.logger.warning(f"Streaming failed for {dataset_name}: {e}")
                
                # Try to get available configs and use the first one
                try:
                    from datasets import get_dataset_infos
                    infos = get_dataset_infos(dataset_name)
                    configs = list(infos.keys())
                    
                    if configs:
                        # Use the first available config
                        config_name = configs[0]
                        self.logger.info(f"Using config '{config_name}' for {dataset_name}")
                        
                        dataset = load_dataset(
                            dataset_name,
                            config_name,
                            streaming=True,
                            cache_dir=str(self.cache_dir)
                        )
                        self.logger.info(f"✅ Downloaded {dataset_name} with config '{config_name}' (streaming mode)")
                        return dataset
                    else:
                        raise ValueError("No configs available")
                        
                except Exception as e2:
                    self.logger.warning(f"Config-based loading failed for {dataset_name}: {e2}")
                    
                    # Fallback to non-streaming with sample limit
                    if max_samples:
                        # For very large datasets, we might need to use a subset
                        dataset = load_dataset(
                            dataset_name,
                            cache_dir=str(self.cache_dir)
                        )
                        # Get first split and limit samples
                        split_name = list(dataset.keys())[0]
                        if len(dataset[split_name]) > max_samples:
                            dataset[split_name] = dataset[split_name].select(range(max_samples))
                        self.logger.info(f"✅ Downloaded {dataset_name} (limited to {max_samples} samples)")
                    else:
                        dataset = load_dataset(
                            dataset_name,
                            cache_dir=str(self.cache_dir)
                        )
                        self.logger.info(f"✅ Downloaded {dataset_name} (full dataset)")
                    
                    return dataset
                
        except Exception as e:
            self.logger.error(f"❌ Failed to download {dataset_name}: {e}")
            raise
    
    def download_all_datasets(self, max_samples_per_dataset: int = 1000):
        """Download all datasets."""
        self.logger.info("🚀 Starting download of all datasets")
        self.logger.info(f"📊 Max samples per dataset: {max_samples_per_dataset}")
        
        downloaded = []
        failed = []
        
        for dataset_name in self.datasets.keys():
            try:
                dataset = self.download_dataset(dataset_name, max_samples_per_dataset)
                downloaded.append(dataset_name)
            except Exception as e:
                self.logger.error(f"Failed to download {dataset_name}: {e}")
                failed.append(dataset_name)
        
        self.logger.info("📊 Download Summary:")
        self.logger.info(f"✅ Successfully downloaded: {len(downloaded)}")
        self.logger.info(f"❌ Failed downloads: {len(failed)}")
        
        if downloaded:
            self.logger.info("Downloaded datasets:")
            for name in downloaded:
                self.logger.info(f"  - {name}")
        
        if failed:
            self.logger.info("Failed datasets:")
            for name in failed:
                self.logger.info(f"  - {name}")
    
    def get_dataset_info(self, dataset_name: str):
        """Get information about a dataset."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets not available")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        try:
            # Load dataset info without downloading
            from datasets import get_dataset_infos
            infos = get_dataset_infos(dataset_name)
            
            print(f"📊 Dataset Information: {dataset_name}")
            print("=" * 50)
            
            for split_name, info in infos.items():
                print(f"Split: {split_name}")
                print(f"  Features: {info.features}")
                print(f"  Num examples: {info.splits.total_num_examples}")
                print(f"  Size in bytes: {info.splits.total_num_bytes}")
                print()
                
        except Exception as e:
            print(f"❌ Could not get info for {dataset_name}: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download pretraining datasets")
    parser.add_argument("--action", type=str, choices=["list", "download", "info"], 
                       default="list", help="Action to perform")
    parser.add_argument("--dataset", type=str, help="Dataset name (for download/info)")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum samples to download per dataset")
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                       help="Cache directory for datasets")
    
    args = parser.parse_args()
    
    if not DATASETS_AVAILABLE:
        print("❌ Required packages not available. Install with:")
        print("pip install datasets")
        return
    
    downloader = DatasetDownloader(args.cache_dir)
    
    if args.action == "list":
        downloader.list_datasets()
    elif args.action == "download":
        if args.dataset:
            downloader.download_dataset(args.dataset, args.max_samples)
        else:
            downloader.download_all_datasets(args.max_samples)
    elif args.action == "info":
        if not args.dataset:
            print("❌ Please specify --dataset for info action")
            return
        downloader.get_dataset_info(args.dataset)

if __name__ == "__main__":
    main()

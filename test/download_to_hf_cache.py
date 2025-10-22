#!/usr/bin/env python3
"""
Download datasets directly to HuggingFace cache using the datasets library
"""
import os
from datasets import load_dataset
import time

def download_dataset_to_cache(dataset_name, subset=None, max_samples=None, split=None):
    """Download a dataset to HuggingFace cache"""
    print(f"\n{'='*60}")
    print(f"📥 Downloading {dataset_name}")
    if subset:
        print(f"   Subset: {subset}")
    if split:
        print(f"   Split: {split}")
    print(f"{'='*60}")
    
    try:
        # Use custom split if provided, otherwise use 'train'
        split_to_use = split if split else 'train'
        
        if subset:
            print(f"Loading {dataset_name} with subset {subset}...")
            dataset = load_dataset(
                dataset_name, 
                subset, 
                split=split_to_use,
                download_mode="reuse_dataset_if_exists"
            )
        else:
            print(f"Loading {dataset_name} without subset...")
            dataset = load_dataset(
                dataset_name, 
                split=split_to_use,
                download_mode="reuse_dataset_if_exists"
            )
        
        print(f"✅ Successfully loaded {dataset_name}")
        print(f"   📊 Dataset size: {len(dataset):,} samples")
        print(f"   📋 Features: {list(dataset.features.keys())}")
        
        # Take a small sample to verify it works
        if max_samples and len(dataset) > max_samples:
            print(f"   🔍 Testing with first {max_samples} samples...")
            sample = dataset.select(range(min(max_samples, len(dataset))))
            print(f"   ✅ Sample loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {e}")
        return False

def download_to_hf_cache():
    """Download datasets directly to HuggingFace cache"""
    
    print("🚀 Downloading datasets to HuggingFace cache...")
    print("📁 Datasets will be stored in ~/.cache/huggingface/datasets/")
    
    # Dataset configurations
    datasets = [
        {
            "name": "HuggingFaceFW/fineweb",
            "subset": "sample-10BT",
            "max_samples": 1000  # Test with small sample, downloads full subset
        },
        {
            "name": "HuggingFaceTB/finemath",
            "subset": "finemath-3plus",
            "max_samples": 1000  # Test with small sample, downloads full subset
        },
        {
            "name": "mlfoundations/dclm-baseline-1.0",
            "subset": None,
            "max_samples": 1000,  # Test with small sample
            "split": "train[:5000000]"  # Limit to 5M samples , ,
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "subset": "auto_math_text",
            "max_samples": 1000  # Test with small sample, downloads full subset
        }
    ]
    
    successful_downloads = []
    
    for dataset_info in datasets:
        success = download_dataset_to_cache(
            dataset_info["name"],
            dataset_info["subset"],
            dataset_info["max_samples"],
            dataset_info.get("split")  # Pass split if it exists
        )
        
        if success:
            successful_downloads.append(dataset_info["name"])
        
        # Small delay between downloads
        time.sleep(2)
    
    print(f"\n🎉 Download Summary:")
    print(f"✅ Successful: {successful_downloads}")
    print(f"📁 All datasets cached in: ~/.cache/huggingface/datasets/")
    
    if successful_downloads:
        print(f"\n💡 Your training script will now use these cached datasets!")
        print(f"   The datasets are ready for training.")
    
    # Show cache info
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        print(f"\n📊 Cache directory contents:")
        for item in sorted(os.listdir(cache_dir)):
            if not item.startswith('.'):
                print(f"   📁 {item}")

if __name__ == "__main__":
    try:
        download_to_hf_cache()
    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

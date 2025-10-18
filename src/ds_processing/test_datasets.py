#!/usr/bin/env python3
"""
Test script for dataset processing functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_dataset_downloader():
    """Test the dataset downloader."""
    print("🧪 Testing Dataset Downloader")
    print("=" * 40)
    
    try:
        from download_datasets import DatasetDownloader
        
        downloader = DatasetDownloader()
        downloader.list_datasets()
        
        print("✅ Dataset downloader test passed")
        return True
    except Exception as e:
        print(f"❌ Dataset downloader test failed: {e}")
        return False

def test_dataset_processor():
    """Test the dataset processor."""
    print("\n🧪 Testing Dataset Processor")
    print("=" * 40)
    
    try:
        from dataset_processor import DatasetProcessor
        
        processor = DatasetProcessor()
        print(f"✅ Loaded {len(processor.dataset_configs)} dataset configurations")
        
        for config in processor.dataset_configs:
            print(f"  - {config.name}: {config.percentage}%")
        
        print("✅ Dataset processor test passed")
        return True
    except Exception as e:
        print(f"❌ Dataset processor test failed: {e}")
        return False

def test_data_loader():
    """Test the data loader."""
    print("\n🧪 Testing Data Loader")
    print("=" * 40)
    
    try:
        from pretraining_data_loader import PretrainingDataLoader
        
        loader = PretrainingDataLoader()
        print("✅ Data loader initialized")
        
        print("✅ Data loader test passed")
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Dataset Processing Components")
    print("=" * 50)
    
    tests = [
        test_dataset_downloader,
        test_dataset_processor,
        test_data_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()

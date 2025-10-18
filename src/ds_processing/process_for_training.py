#!/usr/bin/env python3
"""
Process datasets for training integration.

This script processes Hugging Face datasets and prepares them for use
with the training scripts.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataset_processor import DatasetProcessor
from pretraining_data_loader import PretrainingDataLoader

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process datasets for training")
    parser.add_argument("--total_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--output_filename", type=str, default="pretraining_data",
                       help="Output filename (without extension)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("🚀 Processing datasets for training")
    print(f"📊 Target samples: {args.total_samples}")
    print(f"📁 Output filename: {args.output_filename}")
    
    # Process datasets
    processor = DatasetProcessor(args.config)
    processor.process_all_datasets(
        total_samples=args.total_samples,
        output_filename=args.output_filename
    )
    
    # Test loading the processed data
    print("\n🧪 Testing data loading...")
    loader = PretrainingDataLoader(args.config)
    
    output_path = Path(processor.processing_config.processed_data_dir) / f"{args.output_filename}.{processor.processing_config.save_format}"
    
    if output_path.exists():
        train_data, val_data = loader.load_pretraining_data(str(output_path))
        print(f"✅ Successfully loaded processed data:")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Validation samples: {len(val_data)}")
        
        if train_data:
            sample = train_data[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Sample length: {sample['length']} tokens")
    else:
        print(f"❌ Output file not found: {output_path}")
    
    print("\n✅ Dataset processing complete!")
    print(f"📁 Processed data saved to: {output_path}")

if __name__ == "__main__":
    main()

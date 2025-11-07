#!/usr/bin/env python3
"""
Create evaluation data file for perplexity and entropy metrics.

This script can create eval_data.jsonl from:
1. HuggingFace datasets (same as training)
2. Existing processed data
3. Custom text files

Usage:
    # From HuggingFace datasets (recommended - matches training data)
    python create_eval_data.py --source hf --config configs/config_production.yaml --stage stage1 --output data/eval_data.jsonl --num_samples 1000
    
    # From existing processed data
    python create_eval_data.py --source processed --input data/processed/training_data.jsonl --output data/eval_data.jsonl --num_samples 1000
    
    # From custom text file
    python create_eval_data.py --source text --input my_texts.txt --output data/eval_data.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. HuggingFace dataset source will not work.")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: yaml library not available. Config-based loading will not work.")


def create_from_specific_datasets(dataset_names: List[str], num_samples: int = 1000, split: str = "test") -> List[Dict[str, Any]]:
    """Create eval data from specific HuggingFace datasets."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library required for HuggingFace dataset source")
    
    eval_texts = []
    samples_per_dataset = max(1, num_samples // len(dataset_names))
    
    print(f"📥 Loading {num_samples} samples from {len(dataset_names)} datasets...")
    print(f"   ({samples_per_dataset} samples per dataset, split: {split})")
    
    for dataset_name in dataset_names:
        try:
            print(f"   Loading: {dataset_name}")
            
            # Load dataset - try test split first, then validation, then train
            # Some datasets require a config name (e.g., gsm8k needs 'main')
            dataset = None
            config_name = None
            
            # Try with config name first (for datasets like gsm8k)
            if dataset_name == "openai/gsm8k":
                config_name = "main"
            
            for split_name in [split, "test", "validation", "val", "train"]:
                try:
                    if config_name:
                        dataset = load_dataset(dataset_name, config_name, split=split_name, streaming=False)
                    else:
                        dataset = load_dataset(dataset_name, split=split_name, streaming=False)
                    if split_name != split:
                        print(f"      Using split: {split_name} (requested {split} not available)")
                    break
                except Exception as e:
                    # Try without config name if it failed
                    if config_name:
                        try:
                            dataset = load_dataset(dataset_name, split=split_name, streaming=False)
                            config_name = None  # Clear config_name for next iteration
                            if split_name != split:
                                print(f"      Using split: {split_name} (requested {split} not available)")
                            break
                        except:
                            continue
                    continue
            
            if dataset is None:
                print(f"   ⚠️  Could not load {dataset_name} - no suitable split found")
                continue
            
            # Extract text from dataset (handle different formats)
            count = 0
            for item in dataset:
                if count >= samples_per_dataset:
                    break
                
                text = None
                
                # Handle GSM8K format: question + answer
                if "question" in item and "answer" in item:
                    text = item["question"] + "\n\n" + item["answer"]
                
                # Handle MBPP format: text (problem description) + code
                elif "text" in item and "code" in item:
                    text = item["text"] + "\n\n" + item["code"]
                elif "prompt" in item and "code" in item:
                    text = item["prompt"] + "\n\n" + item["code"]
                
                # Generic formats
                elif "text" in item:
                    text = item["text"]
                elif "content" in item:
                    text = item["content"]
                elif "code" in item:
                    text = item["code"]
                elif "input" in item:
                    # For instruction datasets, combine input and output
                    text = item.get("input", "") + "\n\n" + item.get("output", "")
                elif "question" in item:
                    # For Q&A datasets, combine question and answer
                    text = item.get("question", "") + "\n\n" + item.get("answer", "")
                else:
                    # Try to find any string field
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 50:  # Reasonable length
                            text = value
                            break
                
                if text and len(text.strip()) > 10:  # Minimum length
                    eval_texts.append({"text": text.strip()})
                    count += 1
            
            print(f"   ✅ Loaded {count} samples from {dataset_name}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to load {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Total: {len(eval_texts)} evaluation samples")
    return eval_texts


def create_from_huggingface(config_path: str, stage: str, num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create eval data from HuggingFace datasets (same as training)."""
    if not DATASETS_AVAILABLE or not YAML_AVAILABLE:
        raise RuntimeError("datasets and yaml libraries required for HuggingFace dataset source")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset configuration
    training_stages = config.get("training_stages", {})
    stage_config = training_stages.get(stage, {})
    data_config = stage_config.get("data", {})
    pretraining_datasets = data_config.get("pretraining_datasets", {})
    
    if not pretraining_datasets:
        raise ValueError(f"No pretraining_datasets found in config for stage {stage}")
    
    dataset_names = list(pretraining_datasets.keys())
    return create_from_specific_datasets(dataset_names, num_samples, split="train")


def create_from_processed_data(input_path: str, num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create eval data from processed training data."""
    eval_texts = []
    
    print(f"📥 Loading from processed data: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed data file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= num_samples:
                break
            
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    
                    # Extract text - try different formats
                    text = None
                    if "text" in item:
                        text = item["text"]
                    elif "content" in item:
                        text = item["content"]
                    elif "input_ids" in item:
                        # If we have tokenized data, we can't easily get text back
                        # Skip these for now
                        continue
                    else:
                        # Try to find text-like fields
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 50:
                                text = value
                                break
                    
                    if text and len(text.strip()) > 10:
                        eval_texts.append({"text": text.strip()})
                        count += 1
                        
                except json.JSONDecodeError:
                    continue
    
    print(f"✅ Loaded {len(eval_texts)} samples from processed data")
    return eval_texts


def create_from_text_file(input_path: str) -> List[Dict[str, Any]]:
    """Create eval data from a simple text file (one text per line or paragraph)."""
    eval_texts = []
    
    print(f"📥 Loading from text file: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Text file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Try to split by paragraphs (double newline) or lines
        if '\n\n' in content:
            texts = content.split('\n\n')
        else:
            texts = content.split('\n')
        
        for text in texts:
            text = text.strip()
            if len(text) > 10:  # Minimum length
                eval_texts.append({"text": text})
    
    print(f"✅ Loaded {len(eval_texts)} samples from text file")
    return eval_texts


def create_from_validation_split(config_path: str, stage: str, num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Create eval data from validation split of training datasets."""
    if not DATASETS_AVAILABLE or not YAML_AVAILABLE:
        raise RuntimeError("datasets and yaml libraries required for validation split source")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset configuration
    training_stages = config.get("training_stages", {})
    stage_config = training_stages.get(stage, {})
    data_config = stage_config.get("data", {})
    pretraining_datasets = data_config.get("pretraining_datasets", {})
    
    if not pretraining_datasets:
        raise ValueError(f"No pretraining_datasets found in config for stage {stage}")
    
    eval_texts = []
    samples_per_dataset = max(1, num_samples // len(pretraining_datasets))
    
    print(f"📥 Loading {num_samples} samples from validation splits...")
    print(f"   ({samples_per_dataset} samples per dataset)")
    
    for dataset_name, dataset_config in pretraining_datasets.items():
        try:
            subset = dataset_config.get("subset")
            
            # Try validation split first, then test, then train
            for split_name in ["validation", "val", "test", "train"]:
                try:
                    if subset:
                        dataset = load_dataset(dataset_name, subset, split=split_name, streaming=False)
                    else:
                        dataset = load_dataset(dataset_name, split=split_name, streaming=False)
                    
                    print(f"   Loading: {dataset_name} (split: {split_name})")
                    break
                except:
                    continue
            else:
                print(f"   ⚠️  No suitable split found for {dataset_name}")
                continue
            
            # Extract text from dataset
            count = 0
            for item in dataset:
                if count >= samples_per_dataset:
                    break
                
                # Try different field names for text content
                text = None
                if "text" in item:
                    text = item["text"]
                elif "content" in item:
                    text = item["content"]
                elif "code" in item:
                    text = item["code"]
                elif "input" in item:
                    text = item.get("input", "") + " " + item.get("output", "")
                elif "question" in item:
                    text = item.get("question", "") + " " + item.get("answer", "")
                else:
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 50:
                            text = value
                            break
                
                if text and len(text.strip()) > 10:
                    eval_texts.append({"text": text.strip()})
                    count += 1
            
            print(f"   ✅ Loaded {count} samples from {dataset_name}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to load {dataset_name}: {e}")
            continue
    
    print(f"✅ Total: {len(eval_texts)} evaluation samples")
    return eval_texts


def main():
    parser = argparse.ArgumentParser(description="Create evaluation data file for perplexity/entropy metrics")
    parser.add_argument("--source", type=str, choices=["hf", "processed", "text", "val", "specific"], 
                       default="specific", help="Source of evaluation data")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["openai/gsm8k", "google-research-datasets/mbpp"],
                       help="Specific HuggingFace dataset names (for 'specific' source)")
    parser.add_argument("--config", type=str, default="configs/config_production.yaml",
                       help="Config file path (for hf/val sources)")
    parser.add_argument("--stage", type=str, default="stage1",
                       help="Training stage (for hf/val sources)")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use (test, validation, train)")
    parser.add_argument("--input", type=str, default=None,
                       help="Input file path (for processed/text sources)")
    parser.add_argument("--output", type=str, default="data/eval_data.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to extract")
    
    args = parser.parse_args()
    
    # Create eval data based on source
    eval_texts = []
    
    if args.source == "specific":
        # Use specific datasets (default: gsm8k and mbpp)
        eval_texts = create_from_specific_datasets(args.datasets, args.num_samples, args.split)
    
    elif args.source == "hf":
        if not args.config:
            raise ValueError("--config required for HuggingFace dataset source")
        eval_texts = create_from_huggingface(args.config, args.stage, args.num_samples)
    
    elif args.source == "val":
        if not args.config:
            raise ValueError("--config required for validation split source")
        eval_texts = create_from_validation_split(args.config, args.stage, args.num_samples)
    
    elif args.source == "processed":
        if not args.input:
            raise ValueError("--input required for processed data source")
        eval_texts = create_from_processed_data(args.input, args.num_samples)
    
    elif args.source == "text":
        if not args.input:
            raise ValueError("--input required for text file source")
        eval_texts = create_from_text_file(args.input)
    
    if not eval_texts:
        print("❌ No evaluation texts created!")
        return
    
    # Save to JSONL file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving {len(eval_texts)} samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in eval_texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved evaluation data to {output_path}")
    print(f"📊 File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n💡 Usage:")
    print(f"   python eval.py --checkpoint <checkpoint> --eval_data {output_path}")


if __name__ == "__main__":
    main()


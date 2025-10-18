#!/usr/bin/env python3
"""
NeMo ModularModel Evaluation Script

This script evaluates trained models using various performance metrics:
- Perplexity: Measures how well the model predicts text
- Syntax Accuracy: Checks if generated code is syntactically correct
- Model Statistics: Parameter count, model size, etc.
"""

import sys
import os
import argparse
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.nemo.nemo_wrapper import create_modular_model_nemo
from src.nemo.config_loader import create_nemo_config_from_existing

def calculate_perplexity(model, tokenizer, texts, max_length=512):
    """Calculate perplexity on a set of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Create labels (shifted by 1 for next token prediction)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # Calculate loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item() * (input_ids.size(1) - 1)  # Exclude last token
            total_tokens += (input_ids.size(1) - 1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss

def evaluate_code_generation(model, tokenizer, prompts, max_new_tokens=50):
    """Evaluate code generation quality."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Get model's dtype and device
            model_dtype = next(model.parameters()).dtype
            model_device = next(model.parameters()).device
            
            # Move inputs to match model
            input_ids = input_ids.to(device=model_device, dtype=torch.long)
            attention_mask = attention_mask.to(device=model_device, dtype=torch.long)
            
            # Simple generation (using top-k sampling)
            generated_tokens = []
            current_input_ids = input_ids
            current_attention_mask = attention_mask
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                logits = outputs['logits']
                
                # Get last token logits
                last_token_logits = logits[0, -1, :]
                
                # Sample from top 5 tokens
                top_k = 5
                top_indices = torch.topk(last_token_logits, top_k).indices
                top_probs = torch.softmax(last_token_logits[top_indices], dim=-1)
                
                # Sample next token
                sampled_idx = torch.multinomial(top_probs, 1)[0]
                next_token = top_indices[sampled_idx]
                
                # Stop if we hit a special token
                next_token_text = tokenizer.decode([next_token])
                if next_token_text in ['<|endoftext|>', '<|end|>', '\n\n']:
                    break
                
                generated_tokens.append(next_token)
                
                # Update input for next iteration
                new_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                new_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, dtype=torch.long, device=model_device)], dim=1)
                
                current_input_ids = new_input_ids
                current_attention_mask = new_attention_mask
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'full_output': prompt + generated_text
            })
    
    return results

def evaluate_syntax_correctness(generated_code):
    """Check if generated code is syntactically correct."""
    try:
        compile(generated_code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive NeMo ModularModel Evaluation")
    parser.add_argument("--model_config", type=str, default="model_config_tiny",
                       help="Model configuration key")
    parser.add_argument("--stage", type=str, default="stage1",
                       help="Training stage")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--eval_data", type=str, default="data/example_training_data.jsonl",
                       help="Path to evaluation data")
    
    args = parser.parse_args()
    
    print("🧪 Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Load configuration and model
    config = create_nemo_config_from_existing(args.model_config, args.stage)
    
    # Disable mixed precision for evaluation
    config['mixed_precision'] = None
    config['use_flash_attention'] = False
    
    # Create model
    model = create_modular_model_nemo(**config)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    new_key = key[6:]
                else:
                    new_key = key
                model_state_dict[new_key] = value
            
            model.load_state_dict(model_state_dict, strict=False)
            model = model.float()  # Convert to float32
            model.eval()
            print(f"✅ Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"❌ Invalid checkpoint format: {args.checkpoint}")
            return
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Load tokenizer
    tokenizer_path = config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ Loaded tokenizer from {tokenizer_path}")
    
    # Load evaluation data
    eval_texts = []
    if os.path.exists(args.eval_data):
        with open(args.eval_data, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                eval_texts.append(data['text'])
        print(f"✅ Loaded {len(eval_texts)} evaluation texts")
    else:
        print(f"❌ Evaluation data not found: {args.eval_data}")
        return
    
    print("\n📊 Evaluation Results:")
    print("-" * 30)
    
    # 1. Perplexity Evaluation
    print("1. Perplexity Evaluation...")
    perplexity, avg_loss = calculate_perplexity(model, tokenizer, eval_texts)
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    
    # 2. Code Generation Evaluation
    print("\n2. Code Generation Evaluation...")
    test_prompts = [
        "def fibonacci(n):",
        "import numpy as np",
        "class Calculator:",
        "def calculate_sum(a, b):",
        "import pandas as pd"
    ]
    
    generation_results = evaluate_code_generation(model, tokenizer, test_prompts, max_new_tokens=20)
    
    syntax_correct = 0
    total_generated = len(generation_results)
    
    for i, result in enumerate(generation_results):
        print(f"\n   Prompt {i+1}: {result['prompt']}")
        print(f"   Generated: {result['generated']}")
        
        # Check syntax
        is_syntax_correct = evaluate_syntax_correctness(result['full_output'])
        if is_syntax_correct:
            syntax_correct += 1
            print(f"   ✅ Syntax: Correct")
        else:
            print(f"   ❌ Syntax: Incorrect")
    
    syntax_accuracy = (syntax_correct / total_generated) * 100
    print(f"\n   Syntax Accuracy: {syntax_accuracy:.1f}% ({syntax_correct}/{total_generated})")
    
    # 3. Model Statistics
    print("\n3. Model Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # 4. Summary
    print("\n📋 Evaluation Summary:")
    print("-" * 20)
    print(f"✅ Model loaded successfully")
    print(f"✅ Perplexity: {perplexity:.2f}")
    print(f"✅ Syntax Accuracy: {syntax_accuracy:.1f}%")
    print(f"✅ Parameters: {total_params:,}")
    
    if perplexity < 10:
        print("🎯 Good perplexity (model is learning well)")
    elif perplexity < 50:
        print("⚠️  Moderate perplexity (model needs more training)")
    else:
        print("❌ High perplexity (model needs significant improvement)")

if __name__ == "__main__":
    main()

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
import math
from collections import Counter
import re

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nemo.nemo_wrapper import create_modular_model_nemo
try:
    from src.nemo.config_loader import create_nemo_config_from_existing
except ImportError:
    create_nemo_config_from_existing = None

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

def calculate_entropy(logits, temperature=1.0):
    """Calculate entropy of the probability distribution."""
    probs = torch.softmax(logits / temperature, dim=-1)
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return entropy.item()

def calculate_perplexity_from_logits(logits, labels, ignore_index=-100):
    """Calculate perplexity directly from logits and labels."""
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return torch.exp(loss).item(), loss.item()

def calculate_generation_entropy(model, tokenizer, texts, max_length=512, temperature=1.0):
    """Calculate average entropy during text generation."""
    model.eval()
    total_entropy = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Get model's device
            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # Calculate entropy for each position
            for i in range(logits.size(1) - 1):  # Exclude last position
                token_entropy = calculate_entropy(logits[0, i, :], temperature)
                total_entropy += token_entropy
                total_tokens += 1
    
    avg_entropy = total_entropy / total_tokens if total_tokens > 0 else 0
    return avg_entropy

def calculate_repetition_score(generated_text):
    """Calculate repetition score (lower is better)."""
    words = generated_text.lower().split()
    if len(words) < 2:
        return 0.0
    
    # Count word repetitions
    word_counts = Counter(words)
    repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
    
    # Calculate repetition ratio
    total_words = len(words)
    repetition_ratio = repeated_words / total_words if total_words > 0 else 0.0
    
    return repetition_ratio

def calculate_coherence_score(generated_text):
    """Calculate coherence score based on sentence structure and keywords."""
    if not generated_text.strip():
        return 0.0
    
    score = 0.0
    
    # Check for complete sentences
    sentences = re.split(r'[.!?]+', generated_text)
    complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
    if len(sentences) > 1:
        score += min(0.3, complete_sentences / len(sentences))
    
    # Check for programming keywords (for code generation)
    code_keywords = ['def', 'class', 'import', 'return', 'if', 'else', 'for', 'while', 'try', 'except']
    found_keywords = sum(1 for keyword in code_keywords if keyword in generated_text)
    score += min(0.4, found_keywords / len(code_keywords))
    
    # Check for balanced brackets
    brackets = {'(': ')', '[': ']', '{': '}'}
    for open_bracket, close_bracket in brackets.items():
        open_count = generated_text.count(open_bracket)
        close_count = generated_text.count(close_bracket)
        if open_count > 0:
            balance_ratio = min(open_count, close_count) / max(open_count, close_count)
            score += balance_ratio * 0.1
    
    return min(1.0, score)

def calculate_diversity_score(generated_texts):
    """Calculate diversity score across multiple generated texts."""
    if len(generated_texts) < 2:
        return 0.0
    
    # Extract unique words from all texts
    all_words = []
    for text in generated_texts:
        words = text.lower().split()
        all_words.extend(words)
    
    unique_words = set(all_words)
    total_words = len(all_words)
    
    diversity_ratio = len(unique_words) / total_words if total_words > 0 else 0.0
    return diversity_ratio

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
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for entropy calculation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum tokens to generate for quality evaluation")
    
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
    
    # 1.5. Entropy Evaluation
    print("\n1.5. Entropy Evaluation...")
    avg_entropy = calculate_generation_entropy(model, tokenizer, eval_texts, temperature=args.temperature)
    print(f"   Average Entropy: {avg_entropy:.4f}")
    print(f"   Entropy Range: 0.0 (very confident) to {math.log(tokenizer.vocab_size):.2f} (very uncertain)")
    
    # Interpret entropy score
    max_entropy = math.log(tokenizer.vocab_size)
    entropy_percentage = (avg_entropy / max_entropy) * 100
    print(f"   Entropy Level: {entropy_percentage:.1f}% of maximum entropy")
    
    if entropy_percentage < 20:
        print("   🎯 Low entropy: Model is very confident (may be overfitting)")
    elif entropy_percentage < 50:
        print("   ✅ Good entropy: Model has balanced confidence")
    elif entropy_percentage < 80:
        print("   ⚠️  High entropy: Model is uncertain (may need more training)")
    else:
        print("   ❌ Very high entropy: Model is very uncertain")
    
    # 2. Code Generation Evaluation
    print("\n2. Code Generation Evaluation...")
    test_prompts = [
        "def fibonacci(n):",
        "import numpy as np",
        "class Calculator:",
        "def calculate_sum(a, b):",
        "import pandas as pd"
    ]
    
    generation_results = evaluate_code_generation(model, tokenizer, test_prompts, max_new_tokens=args.max_new_tokens)
    
    syntax_correct = 0
    total_generated = len(generation_results)
    repetition_scores = []
    coherence_scores = []
    generated_texts = []
    
    print("\n   Generation Quality Analysis:")
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
        
        # Calculate quality scores
        repetition_score = calculate_repetition_score(result['generated'])
        coherence_score = calculate_coherence_score(result['generated'])
        
        repetition_scores.append(repetition_score)
        coherence_scores.append(coherence_score)
        generated_texts.append(result['generated'])
        
        print(f"   📊 Repetition Score: {repetition_score:.3f} (lower is better)")
        print(f"   📊 Coherence Score: {coherence_score:.3f} (higher is better)")
    
    syntax_accuracy = (syntax_correct / total_generated) * 100
    avg_repetition = np.mean(repetition_scores) if repetition_scores else 0.0
    avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
    diversity_score = calculate_diversity_score(generated_texts)
    
    print(f"\n   📊 Generation Quality Summary:")
    print(f"   Syntax Accuracy: {syntax_accuracy:.1f}% ({syntax_correct}/{total_generated})")
    print(f"   Average Repetition Score: {avg_repetition:.3f} (lower is better)")
    print(f"   Average Coherence Score: {avg_coherence:.3f} (higher is better)")
    print(f"   Diversity Score: {diversity_score:.3f} (higher is better)")
    
    # Quality interpretation
    if avg_repetition < 0.1:
        print("   ✅ Low repetition: Good diversity in generation")
    elif avg_repetition < 0.3:
        print("   ⚠️  Moderate repetition: Some repetitive patterns")
    else:
        print("   ❌ High repetition: Model tends to repeat phrases")
    
    if avg_coherence > 0.7:
        print("   ✅ High coherence: Well-structured output")
    elif avg_coherence > 0.4:
        print("   ⚠️  Moderate coherence: Decent structure")
    else:
        print("   ❌ Low coherence: Poor structure and flow")
    
    if diversity_score > 0.8:
        print("   ✅ High diversity: Model generates varied content")
    elif diversity_score > 0.5:
        print("   ⚠️  Moderate diversity: Some variation in generation")
    else:
        print("   ❌ Low diversity: Model generates similar content")
    
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
    print(f"✅ Average Entropy: {avg_entropy:.4f} ({entropy_percentage:.1f}% of max)")
    print(f"✅ Syntax Accuracy: {syntax_accuracy:.1f}%")
    print(f"✅ Repetition Score: {avg_repetition:.3f}")
    print(f"✅ Coherence Score: {avg_coherence:.3f}")
    print(f"✅ Diversity Score: {diversity_score:.3f}")
    print(f"✅ Parameters: {total_params:,}")
    
    # Overall quality assessment
    print(f"\n🎯 Overall Quality Assessment:")
    
    if perplexity < 10:
        print("🎯 Good perplexity (model is learning well)")
    elif perplexity < 50:
        print("⚠️  Moderate perplexity (model needs more training)")
    else:
        print("❌ High perplexity (model needs significant improvement)")
    
    # Entropy assessment
    if entropy_percentage < 30:
        print("🎯 Good entropy balance (model is confident but not overconfident)")
    elif entropy_percentage < 60:
        print("⚠️  Moderate entropy (model could be more confident)")
    else:
        print("❌ High entropy (model is very uncertain)")
    
    # Generation quality assessment
    quality_score = (syntax_accuracy/100 * 0.3 + (1-avg_repetition) * 0.3 + avg_coherence * 0.2 + diversity_score * 0.2) * 100
    print(f"📊 Overall Generation Quality Score: {quality_score:.1f}/100")
    
    if quality_score > 80:
        print("🏆 Excellent generation quality!")
    elif quality_score > 60:
        print("✅ Good generation quality")
    elif quality_score > 40:
        print("⚠️  Moderate generation quality")
    else:
        print("❌ Poor generation quality - needs improvement")

if __name__ == "__main__":
    main()

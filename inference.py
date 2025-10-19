#!/usr/bin/env python3
"""
NeMo ModularModel Inference Script

This script loads a trained model and generates text completions from prompts.
Supports various model configurations and training stages.
"""

import sys
import os
import argparse
from pathlib import Path
import torch

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nemo.nemo_wrapper import create_modular_model_nemo
try:
    from src.nemo.config_loader import create_nemo_config_from_existing
except ImportError:
    create_nemo_config_from_existing = None
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="NeMo ModularModel Text Generation")
    parser.add_argument("--model_config", type=str, default="model_config_tiny",
                       help="Model configuration key (e.g., model_config_tiny, model_config_1.7B)")
    parser.add_argument("--stage", type=str, default="stage1",
                       help="Training stage (stage0, stage1, stage2)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top tokens to sample from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_nemo_config_from_existing(args.model_config, args.stage)
    
    # Disable mixed precision for inference
    config['mixed_precision'] = None
    config['use_flash_attention'] = False  # Disable flash attention for inference
    
    # Create model
    model = create_modular_model_nemo(**config)
    
    # Load tokenizer first
    tokenizer_path = config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ Loaded tokenizer from {tokenizer_path}")
    
    # Load checkpoint with proper dtype handling
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    new_key = key[6:]
                else:
                    new_key = key
                # Keep the original dtype from checkpoint (bfloat16)
                model_state_dict[new_key] = value
            
            # Load state dict while preserving dtypes
            model.load_state_dict(model_state_dict, strict=False)
            print(f"✅ Loaded checkpoint from {args.checkpoint}")
            
            # Explicitly convert all parameters to float32 to avoid dtype issues
            model = model.float()
            print(f"📊 Model converted to float32")
            
            # Check model dtype after conversion
            sample_param = next(iter(model.parameters()))
            print(f"📊 Model dtype after conversion: {sample_param.dtype}")
            
        else:
            print(f"❌ Invalid checkpoint format: {args.checkpoint}")
            return
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Set model to evaluation mode (keep original dtype)
    model.eval()
    
    # Prepare input with matching dtype
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get model's dtype and device
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    
    # Move inputs to match model's device and ensure proper dtype for token ids
    input_ids = input_ids.to(device=model_device, dtype=torch.long)  # Token IDs should be long
    attention_mask = attention_mask.to(device=model_device, dtype=torch.long)  # Attention mask should be long
    
    print(f"📊 Model device: {model_device}, dtype: {model_dtype}")
    print(f"📊 Input device: {input_ids.device}, dtype: {input_ids.dtype}")
    
    print(f"📝 Prompt: {args.prompt}")
    print("🔄 Running forward pass...")
    
    # Simple forward pass to get logits
    with torch.no_grad():
        try:
            # Get the underlying model
            if hasattr(model, 'modular_model'):
                underlying_model = model.modular_model
            else:
                underlying_model = model
            
            # Forward pass
            outputs = underlying_model(input_ids=input_ids, attention_mask=attention_mask)
            
            print(f"📊 Output type: {type(outputs)}")
            if isinstance(outputs, dict):
                print(f"📊 Output keys: {list(outputs.keys())}")
                if 'logits' in outputs:
                    logits = outputs['logits']
                    print(f"📊 Logits from dict['logits']: {type(logits)}, shape: {logits.shape}")
                else:
                    print(f"❌ No 'logits' key found in output dict")
                    return
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"📊 Logits from outputs.logits: {type(logits)}, shape: {logits.shape}")
            elif isinstance(outputs, tuple):
                logits = outputs[0]
                print(f"📊 Logits from tuple[0]: {type(logits)}, shape: {logits.shape}")
            else:
                logits = outputs
                print(f"📊 Logits direct: {type(logits)}, shape: {logits.shape}")
            
            # Get the last token's logits
            if isinstance(logits, tuple):
                logits = logits[0]
                print(f"📊 Logits after tuple check: {type(logits)}, shape: {logits.shape}")
            
            last_token_logits = logits[0, -1, :]
            print(f"📊 Last token logits shape: {last_token_logits.shape}")
            
            # Get top predictions
            top_indices = torch.topk(last_token_logits, args.top_k).indices
            top_probs = torch.softmax(last_token_logits[top_indices] / args.temperature, dim=-1)
            
            print(f"📊 Top {args.top_k} next token predictions:")
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' (prob: {prob:.4f})")
            
            # Generate continuation using temperature sampling
            print(f"\n🎯 Generated continuation:")
            generated_tokens = []
            current_input_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            
            for step in range(args.max_tokens):
                # Get logits for current sequence
                with torch.no_grad():
                    if hasattr(model, 'modular_model'):
                        outputs = model.modular_model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                    else:
                        outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                    
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    if isinstance(logits, tuple):
                        logits = logits[0]
                
                # Get last token logits and apply temperature
                last_token_logits = logits[0, -1, :] / args.temperature
                
                # Sample from top_k tokens
                top_k_logits = torch.topk(last_token_logits, args.top_k)
                top_k_probs = torch.softmax(top_k_logits.values, dim=-1)
                sampled_idx = torch.multinomial(top_k_probs, 1)[0]
                next_token_id = top_k_logits.indices[sampled_idx]
                
                # Decode and print token
                next_token = tokenizer.decode([next_token_id])
                generated_tokens.append(next_token)
                print(next_token, end="", flush=True)
                
                # Update input for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, dtype=torch.long)], dim=1)
                
                # Truncate if sequence gets too long
                if current_input_ids.shape[1] > 512:
                    current_input_ids = current_input_ids[:, -512:]
                    current_attention_mask = current_attention_mask[:, -512:]
                
                # Stop if we hit a special token
                if next_token in ['<|endoftext|>', '<|end|>', '\n\n']:
                    break
            
            print(f"\n\n✅ Generation complete! Generated {len(generated_tokens)} tokens.")
            print(f"📊 Full generated text: {args.prompt}{''.join(generated_tokens)}")
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

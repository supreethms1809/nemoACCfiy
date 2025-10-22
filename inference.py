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
from torch.cuda.amp import autocast

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
    parser.add_argument("--model_config", type=str, default="model_config_243M",
                       help="Model configuration key (e.g., model_config_243M, model_config_1.8B)")
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
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty (1.0 = no penalty, >1.0 = reduce repetition)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Use mixed precision (FP16) for inference (default: True)")
    parser.add_argument("--no_mixed_precision", action="store_true", default=False,
                       help="Disable mixed precision and use FP32")
    parser.add_argument("--use_cache", action="store_true", default=True,
                       help="Use KV caching for faster generation (default: True)")
    parser.add_argument("--no_cache", action="store_true", default=False,
                       help="Disable KV caching")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference (default: 1)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"📊 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Determine mixed precision usage
    use_mixed_precision = args.mixed_precision and not args.no_mixed_precision and device.type == 'cuda'
    print(f"📊 Mixed precision: {'Enabled (FP16)' if use_mixed_precision else 'Disabled (FP32)'}")
    
    # Determine KV cache usage
    use_kv_cache = args.use_cache and not args.no_cache and device.type == 'cuda'
    print(f"📊 KV Cache: {'Enabled' if use_kv_cache else 'Disabled'}")
    
    if use_kv_cache:
        print(f"📊 Batch size: {args.batch_size}")
    
    # Load configuration
    config = create_nemo_config_from_existing(args.model_config, args.stage)
    
    # Set mixed precision based on arguments and device capability
    config['mixed_precision'] = "bf16" if use_mixed_precision else None
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
            
            # Move model to device first
            model = model.to(device)
            print(f"📊 Model moved to {device}")
            
            # Convert to appropriate precision based on mixed precision setting
            if use_mixed_precision:
                model = model.half()  # Convert to FP16 for mixed precision
                print(f"📊 Model converted to FP16 (mixed precision)")
            else:
                model = model.float()  # Convert to FP32
                print(f"📊 Model converted to FP32")
            
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
    
    # Move inputs to device and ensure proper dtype for token ids
    input_ids = input_ids.to(device=device, dtype=torch.long)  # Token IDs should be long
    attention_mask = attention_mask.to(device=device, dtype=torch.long)  # Attention mask should be long
    
    # For mixed precision, we'll use autocast context during forward passes
    if use_mixed_precision:
        print(f"📊 Using autocast for mixed precision inference")
    
    print(f"📊 Model device: {device}")
    print(f"📊 Input device: {input_ids.device}, dtype: {input_ids.dtype}")
    
    print(f"📝 Prompt: {args.prompt}")
    print(f"🎯 Generation parameters: max_tokens={args.max_tokens}, temp={args.temperature}, top_k={args.top_k}, repetition_penalty={args.repetition_penalty}")
    
    # Use optimized generation if model supports it and KV cache is enabled
    if use_kv_cache and hasattr(model, 'modular_model') and hasattr(model.modular_model, 'generate'):
        print("🔄 Running optimized inference with KV caching...")
        generate_optimized(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)
    else:
        print("🔄 Running standard inference...")
        generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)

def generate_optimized(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision):
    """Optimized generation using model's built-in generate method with KV caching."""
    try:
        # Use the model's built-in generate method
        if use_mixed_precision:
            with autocast():
                generated_ids = model.modular_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
        else:
            generated_ids = model.modular_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt_text):]
        
        print(f"🎯 Generated text: {new_text}")
        print(f"\n✅ Optimized generation complete! Generated {len(new_text.split())} words.")
        
    except Exception as e:
        print(f"❌ Optimized generation failed: {e}")
        print("🔄 Falling back to standard generation...")
        generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)

def generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision):
    """Standard generation method (fallback)."""
    print("🔄 Running standard inference...")
    
    # Simple forward pass to get logits
    with torch.no_grad():
        try:
            # Get the underlying model
            if hasattr(model, 'modular_model'):
                underlying_model = model.modular_model
            else:
                underlying_model = model
            
            # Forward pass with autocast for mixed precision
            if use_mixed_precision:
                with autocast():
                    outputs = underlying_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
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
            
            # Generate continuation using temperature sampling with repetition penalty
            print(f"\n🎯 Generated continuation:")
            generated_tokens = []
            generated_token_ids = []
            current_input_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            
            for step in range(args.max_tokens):
                # Get logits for current sequence
                with torch.no_grad():
                    if use_mixed_precision:
                        with autocast():
                            if hasattr(model, 'modular_model'):
                                outputs = model.modular_model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                            else:
                                outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                    else:
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
                
                # Get last token logits
                last_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty
                if args.repetition_penalty != 1.0 and len(generated_token_ids) > 0:
                    for token_id in set(generated_token_ids):
                        if last_token_logits[token_id] < 0:
                            last_token_logits[token_id] *= args.repetition_penalty
                        else:
                            last_token_logits[token_id] /= args.repetition_penalty
                
                # Apply temperature
                last_token_logits = last_token_logits / args.temperature
                
                # Sample from top_k tokens
                top_k_logits = torch.topk(last_token_logits, args.top_k)
                top_k_probs = torch.softmax(top_k_logits.values, dim=-1)
                sampled_idx = torch.multinomial(top_k_probs, 1)[0]
                next_token_id = top_k_logits.indices[sampled_idx]
                
                # Decode and print token
                next_token = tokenizer.decode([next_token_id])
                generated_tokens.append(next_token)
                generated_token_ids.append(next_token_id.item())
                print(next_token, end="", flush=True)
                
                # Update input for next iteration
                next_token_id_tensor = next_token_id.unsqueeze(0).unsqueeze(0).to(device)
                current_input_ids = torch.cat([current_input_ids, next_token_id_tensor], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
                
                # Truncate if sequence gets too long (keep context window manageable)
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

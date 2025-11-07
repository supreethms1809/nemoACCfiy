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
import json
from typing import List, Dict, Any

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

def load_prompts_from_file(prompts_file: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Try common keys: 'prompt', 'text', 'input'
                prompt = item.get('prompt') or item.get('text') or item.get('input')
                if prompt:
                    prompts.append(prompt)
            elif isinstance(item, str):
                prompts.append(item)
    elif isinstance(data, dict):
        # If it's a dict, try to find prompts in common keys
        if 'prompts' in data:
            prompts = data['prompts']
        elif 'prompt' in data:
            prompts = [data['prompt']]
    
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}. Expected a list of objects with 'prompt' or 'text' keys, or a list of strings.")
    
    return prompts

def run_single_inference(model, tokenizer, prompt: str, args, device, use_mixed_precision, use_kv_cache) -> str:
    """Run inference on a single prompt and return the generated text."""
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.long)
    
    # Check if model has generate method
    has_generate = False
    actual_model = None
    if hasattr(model, 'modular_model') and hasattr(model.modular_model, 'model') and hasattr(model.modular_model.model, 'generate'):
        actual_model = model.modular_model.model
        has_generate = True
    elif hasattr(model, 'model') and hasattr(model.model, 'generate'):
        actual_model = model.model
        has_generate = True
    elif hasattr(model, 'generate'):
        if hasattr(model, 'modular_model'):
            if hasattr(model.modular_model, 'model'):
                actual_model = model.modular_model.model
            else:
                actual_model = model.modular_model
        else:
            actual_model = model
        has_generate = True
    
    # Use optimized generation if available
    if use_kv_cache and has_generate:
        if hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'generate'):
            actual_model = actual_model.decoder
        
        do_sample = not args.greedy
        temperature = 1.0 if args.greedy else args.temperature
        top_k = 1 if args.greedy else (args.top_k if args.top_k > 0 else 50)
        
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': args.max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': 0.95,
            'do_sample': do_sample,
            'repetition_penalty': args.repetition_penalty,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        }
        
        with torch.no_grad():
            if use_mixed_precision:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    generated_ids = actual_model.generate(**generate_kwargs)
            else:
                generated_ids = actual_model.generate(**generate_kwargs)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        if generated_text.startswith(prompt_text):
            new_text = generated_text[len(prompt_text):]
        else:
            new_text = generated_text
        
        return new_text.strip()
    else:
        # Fallback to standard generation
        generated_tokens = []
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        
        with torch.no_grad():
            for step in range(args.max_tokens):
                if use_mixed_precision:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                else:
                    outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                
                # Extract logits
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs)
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                last_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty
                if args.repetition_penalty != 1.0 and len(generated_tokens) > 0:
                    unique_generated = set(generated_tokens)
                    for token_id in unique_generated:
                        if last_token_logits[token_id] < 0:
                            last_token_logits[token_id] *= args.repetition_penalty
                        else:
                            last_token_logits[token_id] /= args.repetition_penalty
                
                # Apply temperature
                if args.temperature != 1.0:
                    last_token_logits = last_token_logits / args.temperature
                
                # Sample from top_k tokens
                top_k_logits, top_k_indices = torch.topk(last_token_logits, min(args.top_k, last_token_logits.size(-1)))
                top_k_probs = torch.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(top_k_probs, 1)[0]
                next_token_id = top_k_indices[sampled_idx]
                
                generated_tokens.append(next_token_id.item())
                
                # Update input for next iteration
                next_token_id_tensor = next_token_id.unsqueeze(0).unsqueeze(0).to(device)
                current_input_ids = torch.cat([current_input_ids, next_token_id_tensor], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
                
                # Truncate if sequence gets too long
                if current_input_ids.shape[1] > 512:
                    current_input_ids = current_input_ids[:, -512:]
                    current_attention_mask = current_attention_mask[:, -512:]
                
                # Stop if we hit a special token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text.strip()

def main():
    parser = argparse.ArgumentParser(description="NeMo ModularModel Text Generation")
    parser.add_argument("--model_config", type=str, default="model_config_243M",
                       help="Model configuration key (e.g., model_config_243M, model_config_1.8B)")
    parser.add_argument("--stage", type=str, default="stage1",
                       help="Training stage (stage0, stage1, stage2)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Input prompt for text generation (required if --prompts_file not provided)")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="Path to JSON file containing prompts to run inference on")
    parser.add_argument("--prompt_file", type=str, default=None,
                       dest="prompts_file", help="Alias for --prompts_file")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to output JSON file to save prompts and generations (required when using --prompts_file)")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (0.1-2.0, use 0.0 for greedy decoding)")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Number of top tokens to sample from (default: 50, use 0 for no limit)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty (1.0 = no penalty, >1.0 = reduce repetition)")
    parser.add_argument("--greedy", action="store_true", default=False,
                       help="Use greedy decoding instead of sampling (temperature=0, top_k=1)")
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
    
    # Handle alias: if --prompt_file was used, it sets prompts_file via dest
    # But we need to check if it was actually provided
    # Since argparse doesn't easily let us check which argument was used,
    # we'll just validate based on the final value
    
    # Validate arguments
    if args.prompts_file is None and args.prompt is None:
        parser.error("Either --prompt or --prompts_file (or --prompt_file) must be provided")
    if args.prompts_file is not None and args.output_file is None:
        parser.error("--output_file is required when using --prompts_file or --prompt_file")
    
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
    # Echo weight tying resolved from config for transparency
    if 'tie_weights' in config:
        print(f"📊 Weight tying (from config): {'Enabled' if config['tie_weights'] else 'Disabled'}")
    else:
        print("📊 Weight tying: not specified in config (will use model default)")
    
    # Set mixed precision based on arguments and device capability
    config['mixed_precision'] = "bf16" if use_mixed_precision else None
    # Keep flash attention enabled to match training (unless explicitly disabled)
    # Flash attention should produce same results as standard attention, just faster
    if not hasattr(args, 'disable_flash_attention') or not args.disable_flash_attention:
        config['use_flash_attention'] = True  # Match training configuration
    
    # Create model
    model = create_modular_model_nemo(**config)
    
    # Load tokenizer first
    tokenizer_path = config.get("tokenizer_path", "tokenizers/qwen3-coder-30b-a3b-instruct-custom")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ Loaded tokenizer from {tokenizer_path}")
    
    # Load checkpoint with proper dtype handling
    if os.path.exists(args.checkpoint):
        # PyTorch 2.6+ defaults to weights_only=True for security, but checkpoints may contain
        # tokenizer objects (e.g., Qwen2TokenizerFast) which require weights_only=False
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            # Debug: Print first few keys to understand structure
            checkpoint_keys = list(checkpoint['state_dict'].keys())[:10]
            print(f"📊 Sample checkpoint keys: {checkpoint_keys}")
            
            # Get model's expected keys
            model_keys = list(model.state_dict().keys())[:10]
            print(f"📊 Sample model keys: {model_keys}")
            
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                # Handle different checkpoint key formats
                # PyTorch Lightning saves with "model." prefix for the wrapped model
                # Structure: model.modular_model.model.decoder.embed_tokens.weight (for ModularModelNeMoWrapper)
                #            model.model.decoder.embed_tokens.weight (for DecoderOnlyModel)
                new_key = key
                
                # Remove "model." prefix if present (PyTorch Lightning adds this)
                if new_key.startswith('model.'):
                    new_key = new_key[6:]  # Remove "model." -> modular_model.model.decoder...
                
                # Now new_key should be: modular_model.model.decoder... or model.decoder...
                # The model expects: modular_model.model.decoder... (no further stripping needed!)
                # So we just use new_key as-is
                
                # Keep the original dtype from checkpoint (bfloat16)
                model_state_dict[new_key] = value
            
            # Load state dict while preserving dtypes
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys (first 10): {missing_keys[:10]}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys (first 10): {unexpected_keys[:10]}")
            
            # Check if we actually loaded any weights
            loaded_params = sum(1 for k in model_state_dict.keys() if k in model.state_dict())
            total_params = len(model.state_dict())
            print(f"📊 Loaded {loaded_params}/{total_params} parameters from checkpoint")
            
            print(f"✅ Loaded checkpoint from {args.checkpoint}")
            
            # Move model to device first
            model = model.to(device)
            print(f"📊 Model moved to {device}")
            
            # Convert to appropriate precision based on mixed precision setting
            # CRITICAL: Training uses BF16, so inference should use BF16 too (not FP16)
            if use_mixed_precision:
                model = model.to(torch.bfloat16)  # Convert to BF16 to match training precision
                print(f"📊 Model converted to BF16 (mixed precision) - matches training precision")
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
    
    # Set model to evaluation mode (CRITICAL: ensure all submodules are in eval mode)
    model.eval()
    # Also ensure nested modules are in eval mode to disable dropout
    if hasattr(model, 'modular_model'):
        model.modular_model.eval()
        if hasattr(model.modular_model, 'model'):
            model.modular_model.model.eval()
            # Also set decoder to eval mode if it exists
            if hasattr(model.modular_model.model, 'decoder'):
                model.modular_model.model.decoder.eval()
    
    print(f"📊 Model is in eval mode: {not model.training}")
    if hasattr(model, 'modular_model') and hasattr(model.modular_model, 'model'):
        print(f"📊 Underlying model is in eval mode: {not model.modular_model.model.training}")
    
    # Handle batch processing from prompts file
    if args.prompts_file:
        print(f"\n📂 Loading prompts from {args.prompts_file}...")
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"✅ Loaded {len(prompts)} prompts")
        
        print(f"\n🎯 Generation parameters: max_tokens={args.max_tokens}, temp={args.temperature}, top_k={args.top_k}, repetition_penalty={args.repetition_penalty}")
        print(f"🔄 Running inference on {len(prompts)} prompts...\n")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Processing prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            try:
                generated_text = run_single_inference(model, tokenizer, prompt, args, device, use_mixed_precision, use_kv_cache)
                results.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "prompt_index": i
                })
                print(f"✅ Generated {len(generated_text)} characters")
            except Exception as e:
                print(f"❌ Error processing prompt {i}: {e}")
                results.append({
                    "prompt": prompt,
                    "generated_text": "",
                    "error": str(e),
                    "prompt_index": i
                })
        
        # Save results to output file
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(results)} results to {output_path}")
        print(f"📊 Successfully processed: {sum(1 for r in results if 'error' not in r)}/{len(results)} prompts")
        return
    
    # Single prompt mode (original behavior)
    if args.prompt is None:
        parser.error("--prompt is required when not using --prompts_file")
    
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
    
    # Test forward pass first to verify model is working
    print("🔍 Testing forward pass to verify model is working...")
    with torch.no_grad():
        # CRITICAL FIX: Use wrapper forward pass (same as training) instead of bypassing
        # This ensures we test the exact same path used during training
        
        if use_mixed_precision:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Use wrapper forward pass (same as training)
                test_output = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # Use wrapper forward pass (same as training)
            test_output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract logits from wrapper output
        if isinstance(test_output, dict):
            test_logits = test_output.get('logits', test_output)
        elif hasattr(test_output, 'logits'):
            test_logits = test_output.logits
        else:
            test_logits = test_output
        
        if isinstance(test_logits, tuple):
            test_logits = test_logits[0]
        
        print(f"✅ Forward pass successful! Logits shape: {test_logits.shape}")
        print(f"📊 Logits dtype: {test_logits.dtype}, device: {test_logits.device}")
        print(f"📊 Logits range: min={test_logits.min().item():.2f}, max={test_logits.max().item():.2f}, mean={test_logits.mean().item():.2f}")
        
        # Test greedy decoding on the prompt to verify model is working
        print("\n🔍 Testing greedy decoding on prompt to verify model quality...")
        last_token_logits = test_logits[0, -1, :]
        greedy_token_id = torch.argmax(last_token_logits).item()
        greedy_token = tokenizer.decode([greedy_token_id])
        print(f"📊 Greedy next token: '{greedy_token}' (token_id: {greedy_token_id})")
        
        # Check top 5 predictions
        top_5_logits, top_5_indices = torch.topk(last_token_logits, 5)
        top_5_probs = torch.softmax(top_5_logits, dim=-1)
        print(f"📊 Top 5 next token predictions:")
        for i, (idx, prob) in enumerate(zip(top_5_indices, top_5_probs)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' (prob: {prob.item():.4f}, logit: {top_5_logits[i].item():.2f})")
    
    # Use optimized generation if model supports it and KV cache is enabled
    # Check if model has generate method (either directly or through model attribute)
    has_generate = False
    if hasattr(model, 'generate'):
        has_generate = True
        print("✅ Model has generate() method")
    elif hasattr(model, 'modular_model') and hasattr(model.modular_model, 'model') and hasattr(model.modular_model.model, 'generate'):
        has_generate = True
        print("✅ Model.modular_model.model has generate() method")
    elif hasattr(model, 'model') and hasattr(model.model, 'generate'):
        has_generate = True
        print("✅ Model.model has generate() method")
    elif hasattr(model, 'modular_model') and hasattr(model.modular_model, 'generate'):
        has_generate = True
        print("✅ Model.modular_model has generate() method")
    
    # Try optimized generation first (uses model's built-in generate method)
    # This is more reliable because it uses the same generation logic as training
    if use_kv_cache and has_generate:
        print("🔄 Running optimized inference with KV caching (using model's generate method)...")
        try:
            generate_optimized(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)
        except Exception as e:
            print(f"⚠️ Optimized generation failed: {e}")
            print("🔄 Falling back to standard generation...")
            generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)
    else:
        print("🔄 Running standard inference...")
        generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)

def generate_optimized(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision):
    """Optimized generation using model's built-in generate method with KV caching."""
    try:
        # Get the actual model that has generate() method
        # Handle different model structures: ModularModelNeMo wrapper, etc.
        # Structure: ModularModelNeMoWrapper -> modular_model (ModularModelNeMo) -> model (DecoderOnlyModel) -> decoder (LMHeadDecoder)
        # IMPORTANT: We need to access the DecoderOnlyModel directly, not the wrapper
        actual_model = None
        if hasattr(model, 'modular_model') and hasattr(model.modular_model, 'model') and hasattr(model.modular_model.model, 'generate'):
            # ModularModelNeMoWrapper -> modular_model -> model (DecoderOnlyModel)
            actual_model = model.modular_model.model
            print("📊 Using model.modular_model.model.generate() (DecoderOnlyModel)")
        elif hasattr(model, 'model') and hasattr(model.model, 'generate'):
            actual_model = model.model
            print("📊 Using model.model.generate()")
        elif hasattr(model, 'generate'):
            # Check if this is the wrapper - if so, we need to go deeper
            if hasattr(model, 'modular_model'):
                # This is ModularModelNeMoWrapper, go to the actual model
                if hasattr(model.modular_model, 'model'):
                    actual_model = model.modular_model.model
                    print("📊 Using model.modular_model.model.generate() (via wrapper)")
                else:
                    actual_model = model.modular_model
                    print("📊 Using model.modular_model.generate()")
            else:
                actual_model = model
                print("📊 Using model.generate()")
        else:
            print("❌ Could not find generate() method, falling back to standard generation")
            raise AttributeError("No generate() method found")
        
        # Use the model's built-in generate method
        # CRITICAL: Use BF16 autocast to match training precision (not FP16)
        # DecoderOnlyModel.generate() delegates to decoder.generate() which accepts:
        # input_ids, attention_mask, max_new_tokens, temperature, top_p, top_k, do_sample, eos_token_id, pad_token_id, repetition_penalty
        # But DecoderOnlyModel.generate() doesn't accept repetition_penalty, so we need to call decoder.generate() directly
        # OR we can call the decoder's generate method directly if it's available
        if hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'generate'):
            # Call decoder.generate() directly to get access to repetition_penalty
            actual_model = actual_model.decoder
            print("📊 Using decoder.generate() directly (has repetition_penalty support)")
        
        # Use greedy decoding if requested, otherwise use sampling
        if args.greedy:
            do_sample = False
            temperature = 1.0
            top_k = 1
            print("📊 Using greedy decoding (temperature=1.0, top_k=1, do_sample=False)")
        else:
            do_sample = True
            temperature = args.temperature
            top_k = args.top_k if args.top_k > 0 else 50
            print(f"📊 Using sampling (temperature={temperature}, top_k={top_k}, do_sample=True)")
        
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': args.max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': 0.95,  # Default top_p for nucleus sampling
            'do_sample': do_sample,
            'repetition_penalty': args.repetition_penalty,
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        }
        
        if use_mixed_precision:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated_ids = actual_model.generate(**generate_kwargs)
        else:
            generated_ids = actual_model.generate(**generate_kwargs)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Handle case where generated_text might be shorter than prompt_text
        if generated_text.startswith(prompt_text):
            new_text = generated_text[len(prompt_text):]
        else:
            # If prompt doesn't match, just use the generated text
            new_text = generated_text
        
        print(f"\n🎯 Generated text: {new_text}")
        print(f"📊 Generated {len(generated_ids[0]) - len(input_ids[0])} new tokens")
        print(f"✅ Optimized generation complete!")
        
    except Exception as e:
        print(f"❌ Optimized generation failed: {e}")
        print("🔄 Falling back to standard generation...")
        generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision)

def generate_standard(model, input_ids, attention_mask, tokenizer, args, device, use_mixed_precision):
    """Standard generation method (fallback)."""
    print("🔄 Running standard inference...")
    
    # CRITICAL FIX: Use the wrapper's forward pass (same as training) instead of bypassing it
    # This ensures consistent behavior between training and inference
    print("📊 Using wrapper forward pass (same as training path)...")
    
    # Simple forward pass to get logits
    with torch.no_grad():
        try:
            # Forward pass with autocast for mixed precision
            # CRITICAL: Use BF16 autocast to match training precision (not FP16)
            # CRITICAL FIX: Use wrapper forward pass (same as training) instead of bypassing
            if use_mixed_precision:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Use wrapper forward pass (same as training)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # Use wrapper forward pass (same as training)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract logits from wrapper output (wrapper returns dict with 'logits' key)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
                print(f"📊 Logits from dict['logits']: shape={logits.shape}")
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"📊 Logits from outputs.logits: shape={logits.shape}")
            elif isinstance(outputs, tuple):
                logits = outputs[0]
                print(f"📊 Logits from tuple[0]: shape={logits.shape}")
            else:
                logits = outputs
                print(f"📊 Logits direct: shape={logits.shape}")
            
            # Get the last token's logits
            if isinstance(logits, tuple):
                logits = logits[0]
            
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
                    # CRITICAL FIX: Use wrapper forward pass (same as training)
                    # CRITICAL: Use BF16 autocast to match training precision (not FP16)
                    if use_mixed_precision:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            # Use wrapper forward pass
                            outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                    else:
                        # Use wrapper forward pass
                        outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
                    
                    # Extract logits (wrapper returns dict)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs)
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    if isinstance(logits, tuple):
                        logits = logits[0]
                
                # Get last token logits
                last_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty (only penalize tokens that appear in the generated sequence)
                if args.repetition_penalty != 1.0 and len(generated_token_ids) > 0:
                    # Only penalize tokens that have been generated (not the prompt)
                    unique_generated = set(generated_token_ids)
                    for token_id in unique_generated:
                        if last_token_logits[token_id] < 0:
                            last_token_logits[token_id] *= args.repetition_penalty
                        else:
                            last_token_logits[token_id] /= args.repetition_penalty
                
                # Apply temperature
                if args.temperature != 1.0:
                    last_token_logits = last_token_logits / args.temperature
                
                # Sample from top_k tokens
                # Get top_k logits and indices
                top_k_logits, top_k_indices = torch.topk(last_token_logits, min(args.top_k, last_token_logits.size(-1)))
                
                # Apply softmax to get probabilities
                top_k_probs = torch.softmax(top_k_logits, dim=-1)
                
                # Sample from the top_k probabilities
                sampled_idx = torch.multinomial(top_k_probs, 1)[0]
                next_token_id = top_k_indices[sampled_idx]
                
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

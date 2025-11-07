#!/usr/bin/env python3
"""
Inference script for the ACCfiy model.
Supports both stage-based inference and full model inference.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import json
from transformers import AutoTokenizer
from model.embed_decoder_model import ModularModel, create_plan_decoder
from utils.training_utils import setup_logging, load_config, get_gpu_memory_usage

def load_model_checkpoint(checkpoint_path, model, device="auto", use_cpu_offload=False, use_mixed_precision=True):
    """Load model from checkpoint with memory optimization options."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain tokenizer objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Memory optimization strategies
    if use_cpu_offload and device.type == 'cuda':
        # Keep embedder on GPU, move decoder to CPU
        if hasattr(model, 'embedder'):
            model.embedder.to(device)
        if hasattr(model, 'decoder'):
            model.decoder.to('cpu')
        print("📊 CPU offloading enabled: embedder on GPU, decoder on CPU")
    else:
        model.to(device)
    
    # Enable mixed precision for inference
    if use_mixed_precision and device.type == 'cuda':
        model = model.half()  # Convert to FP16
        print("📊 Mixed precision (FP16) enabled for inference")
    
    model.eval()
    
    # Print memory usage
    if device.type == 'cuda':
        memory_stats = get_gpu_memory_usage()
        print(f"📊 GPU Memory after loading model: {memory_stats['memory_allocated_mb']:.1f} MB")
    
    return model

def load_plan_decoder_checkpoint(checkpoint_path, config, tokenizer, device="auto"):
    """Load plan decoder from checkpoint."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create plan decoder with same config and tokenizer as training
    plan_decoder = create_plan_decoder(
        config=config,
        tokenizer=tokenizer,
        use_pretrained=False,  # Use custom model architecture (same as training)
        device=device
    )
    
    # Load checkpoint
    # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain tokenizer objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        # Handle vocabulary size mismatch by resizing embeddings
        state_dict = checkpoint['model_state_dict']
        if 'model.transformer.wte.weight' in state_dict:
            old_vocab_size = state_dict['model.transformer.wte.weight'].shape[0]
            new_vocab_size = config['vocab_size']
            if old_vocab_size != new_vocab_size:
                print(f"📊 Resizing vocabulary from {old_vocab_size} to {new_vocab_size}")
                # The model will automatically handle the resize
        plan_decoder.load_state_dict(state_dict, strict=False)
    else:
        plan_decoder.load_state_dict(checkpoint, strict=False)
    
    plan_decoder.to(device)
    plan_decoder.eval()
    return plan_decoder

def load_decoder_only_checkpoint(checkpoint_path, config, device="auto", use_cpu_offload=False, use_mixed_precision=True):
    """Load decoder-only model from checkpoint (extracted from Stage 1 training)."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create decoder-only model using the decoder config
    from model.embed_decoder_model import LMHeadDecoder
    decoder = LMHeadDecoder(config['decoder_config'], config['vocab_size'])
    
    # Load checkpoint
    # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain tokenizer objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Extract decoder-specific state dict keys
        decoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('decoder.'):
                # Remove 'decoder.' prefix
                new_key = key[8:]  # Remove 'decoder.'
                decoder_state_dict[new_key] = value
            elif key.startswith('lm_head.'):
                # Keep lm_head keys as is
                decoder_state_dict[key] = value
        
        decoder.load_state_dict(decoder_state_dict, strict=False)
    else:
        # Assume the checkpoint is already decoder-only
        decoder.load_state_dict(checkpoint, strict=False)
    
    # Memory optimization strategies
    if use_cpu_offload and device.type == 'cuda':
        # For decoder-only, we can keep it on GPU but with reduced precision
        decoder.to(device)
        print("📊 Decoder-only model loaded on GPU")
    else:
        decoder.to(device)
    
    # Enable mixed precision for inference
    if use_mixed_precision and device.type == 'cuda':
        decoder = decoder.half()  # Convert to FP16
        print("📊 Mixed precision (FP16) enabled for decoder-only inference")
    
    decoder.eval()
    
    # Print memory usage
    if device.type == 'cuda':
        memory_stats = get_gpu_memory_usage()
        print(f"📊 GPU Memory after loading decoder-only model: {memory_stats['memory_allocated_mb']:.1f} MB")
    
    return decoder

def generate_plan(plan_decoder, prompt, tokenizer, max_new_tokens=256, temperature=0.7, 
                 top_p=0.9, top_k=50, do_sample=True):
    """Generate planning sequence from prompt."""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move to device and match model's dtype
        device = next(plan_decoder.parameters()).device
        dtype = next(plan_decoder.parameters()).dtype  # Get model's dtype
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate plan
        with torch.no_grad():
            outputs = plan_decoder.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode plan
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        plan = generated_text[len(prompt):].strip()
        
        return plan
        
    except Exception as e:
        print(f"Error generating plan: {e}")
        return f"Error generating plan: {str(e)}"

def generate_with_modular_model(model, prompt, plan, tokenizer, max_new_tokens=512, 
                               temperature=0.7, top_p=0.9, top_k=50, do_sample=True):
    """Generate response using ModularModel with plan embedding."""
    try:
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
        attention_mask = torch.ones_like(input_ids)
        
        # Tokenize plan for embedding
        embed_input_ids = tokenizer.encode(plan, return_tensors="pt", truncation=True, max_length=4096)
        embed_attention_mask = torch.ones_like(embed_input_ids)
        
        # Move to device and match model's dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype  # Get model's dtype
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device).to(dtype)  # Match model's dtype
        embed_input_ids = embed_input_ids.to(device)
        embed_attention_mask = embed_attention_mask.to(device).to(dtype)  # Match model's dtype
        
        # Generate with ModularModel
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                embed_input_ids=embed_input_ids,
                attention_mask=attention_mask,
                embed_attention_mask=embed_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Remove the original prompt from response
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

def generate_with_decoder_only(decoder, prompt, tokenizer, max_new_tokens=512, 
                              temperature=0.7, top_p=0.9, top_k=50, do_sample=True, repetition_penalty=1.1):
    """Generate response using only the decoder component (Stage 1 style)."""
    try:
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
        attention_mask = torch.ones_like(input_ids)
        
        # Move to device and match model's dtype
        device = next(decoder.parameters()).device
        dtype = next(decoder.parameters()).dtype  # Get model's dtype
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device).to(dtype)  # Match model's dtype
        
        # Generate with decoder only (no cross-attention)
        with torch.no_grad():
            generated_ids = decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Remove the original prompt from response
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating response with decoder only: {e}")
        return f"Error generating response: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="ACCfiy Inference")
    
    # Model paths
    parser.add_argument("--stage0-model-path", type=str, default=None, 
                       help="Path to Stage 0 plan decoder checkpoint")
    parser.add_argument("--stage1-model-path", type=str, default=None, 
                       help="Path to Stage 1 pre-trained model checkpoint")
    parser.add_argument("--stage2-model-path", type=str, default=None, 
                       help="Path to Stage 2 full model checkpoint")
    parser.add_argument("--decoder-only-model-path", type=str, default=None,
                       help="Path to decoder-only model checkpoint (extracted from Stage 1)")
    parser.add_argument("--config-file", type=str, default="src/model_config/config.json",
                       help="Path to model configuration file")
    parser.add_argument("--model-config-key", type=str, default="model_config_89M",
                       help="Model configuration key")
    
    # Tokenizer
    parser.add_argument("--tokenizer-name", type=str, default="Qwen/Qwen3-Coder-480B-A35B-Instruct",
                       help="Tokenizer name or path")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=512, 
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--plan-max-tokens", type=int, default=256, 
                       help="Maximum tokens for plan generation")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=50, 
                       help="Top-k for sampling")
    parser.add_argument("--top-p", type=float, default=0.9, 
                       help="Top-p for sampling")
    parser.add_argument("--do-sample", action="store_true", 
                       help="Enable sampling (default: greedy decoding)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Repetition penalty (1.0 = no penalty, >1.0 = penalty)")
    
    # Memory optimization
    parser.add_argument("--use-cpu-offload", action="store_true",
                       help="Use CPU offloading for large models")
    parser.add_argument("--use-mixed-precision", action="store_true", default=True,
                       help="Use mixed precision (FP16) for inference")
    parser.add_argument("--max-memory", type=str, default=None,
                       help="Maximum GPU memory to use (e.g., '8GB')")
    
    # Input
    parser.add_argument("--prompt", type=str, default="Write a python function to calculate fibonacci numbers",
                       help="Input prompt for generation")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    
    # Output
    parser.add_argument("--output-file", type=str, default=None, 
                       help="Save output to file")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed output including plan")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting ACCfiy inference...")
    
    # Load tokenizer - use the same tokenizer as training
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model configuration from config.json
    logger.info(f"Loading config from {args.config_file} with model config key: {args.model_config_key}")
    config = load_config(args.config_file, args.model_config_key)
    
    # Set GPU memory limit if specified
    if args.max_memory and torch.cuda.is_available():
        import re
        memory_gb = float(re.findall(r'(\d+(?:\.\d+)?)', args.max_memory)[0])
        memory_bytes = int(memory_gb * 1024**3)
        torch.cuda.set_per_process_memory_fraction(memory_bytes / torch.cuda.get_device_properties(0).total_memory)
        logger.info(f"📊 GPU memory limit set to {args.max_memory}")
    
    # Determine inference mode
    if args.stage2_model_path:
        # Full model inference (Stage 2)
        logger.info("Using Stage 2 full model inference")
        model = ModularModel(config)
        model = load_model_checkpoint(args.stage2_model_path, model, 
                                    use_cpu_offload=args.use_cpu_offload,
                                    use_mixed_precision=args.use_mixed_precision)
        
        def generate_response(prompt):
            # For full model, we need a plan first
            if args.stage0_model_path:
                # Use Stage 0 for plan generation
                plan_config = {'vocab_size': len(tokenizer), 'decoder_config': None}
                plan_decoder = load_plan_decoder_checkpoint(args.stage0_model_path, plan_config, tokenizer)
                plan = generate_plan(plan_decoder, prompt, tokenizer, 
                                   max_new_tokens=args.plan_max_tokens,
                                   temperature=args.temperature, top_p=args.top_p, 
                                   top_k=args.top_k, do_sample=args.do_sample)
                if args.verbose:
                    print(f"\n📋 Generated Plan:\n{plan}\n")
            else:
                # Use a simple default plan
                plan = "1. Understand the requirements 2. Break down the problem 3. Implement the solution 4. Test the code"
                if args.verbose:
                    print(f"\n📋 Using Default Plan:\n{plan}\n")
            
            # Generate response with plan
            response = generate_with_modular_model(model, prompt, plan, tokenizer,
                                                 max_new_tokens=args.max_new_tokens,
                                                 temperature=args.temperature, 
                                                 top_p=args.top_p, top_k=args.top_k, 
                                                 do_sample=args.do_sample)
            return response
    
    elif args.decoder_only_model_path:
        # Decoder-only inference (extracted from Stage 1)
        logger.info("Using decoder-only inference (extracted from Stage 1)")
        decoder = load_decoder_only_checkpoint(args.decoder_only_model_path, config,
                                             use_cpu_offload=args.use_cpu_offload,
                                             use_mixed_precision=args.use_mixed_precision)
        
        def generate_response(prompt):
            # Decoder-only generation (no cross-attention, no plan needed)
            response = generate_with_decoder_only(decoder, prompt, tokenizer,
                                                max_new_tokens=args.max_new_tokens,
                                                temperature=args.temperature, 
                                                top_p=args.top_p, top_k=args.top_k, 
                                                do_sample=args.do_sample,
                                                repetition_penalty=args.repetition_penalty)
            return response
    
    elif args.stage1_model_path:
        # Stage 1 inference (decoder-only)
        logger.info("Using Stage 1 decoder-only inference")
        model = ModularModel(config)
        model = load_model_checkpoint(args.stage1_model_path, model,
                                    use_cpu_offload=args.use_cpu_offload,
                                    use_mixed_precision=args.use_mixed_precision)
        
        def generate_response(prompt):
            # Stage 1: decoder-only generation (no cross-attention)
            input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
            attention_mask = torch.ones_like(input_ids)
            
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype  # Get model's dtype
            
            # Note: input_ids should stay as Long (int64) for embedding lookup
            # Only attention_mask needs to match model's dtype
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device).to(dtype)  # Match model's dtype
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    embed_input_ids=None,  # No cross-attention for Stage 1
                    attention_mask=attention_mask,
                    embed_attention_mask=None,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            return response
    
    elif args.stage0_model_path:
        # Stage 0 inference (plan generation only)
        logger.info("Using Stage 0 plan generation inference")
        
        # Use the same config and tokenizer as training
        plan_config = {'vocab_size': len(tokenizer), 'decoder_config': config['decoder_config']}
        plan_decoder = load_plan_decoder_checkpoint(args.stage0_model_path, plan_config, tokenizer)
        
        def generate_response(prompt):
            plan = generate_plan(plan_decoder, prompt, tokenizer, 
                               max_new_tokens=args.plan_max_tokens,
                               temperature=args.temperature, top_p=args.top_p, 
                               top_k=args.top_k, do_sample=args.do_sample)
            return plan
    
    else:
        logger.error("No model checkpoint provided. Use --stage0-model-path, --stage1-model-path, --stage2-model-path, or --decoder-only-model-path")
        return
    
    # Run inference
    if args.interactive:
        print("🤖 ACCfiy Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\n💬 Enter your prompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt.strip():
                    continue
                
                print("\n🔄 Generating...")
                response = generate_response(prompt)
                print(f"\n✨ Response:\n{response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    else:
        # Single inference
        print(f"💬 Prompt: {args.prompt}")
        print("\n🔄 Generating...")
        
        response = generate_response(args.prompt)
        
        print(f"\n✨ Response:\n{response}")
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Prompt: {args.prompt}\n\n")
                f.write(f"Response: {response}\n")
            print(f"\n💾 Output saved to: {args.output_file}")
    
    logger.info("Inference completed!")

if __name__ == "__main__":
    main()
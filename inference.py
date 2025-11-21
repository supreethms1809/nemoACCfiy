#!/usr/bin/env python3
"""
NeMo ModularModel Inference Script

This script loads a trained model and generates text completions from prompts.
Supports various model configurations and training stages:
- stage1: Standard next-token prediction (plain text prompts)
- stage1_inst_SFT: Instruction fine-tuning format (prompts are automatically formatted with <|instruction|> and <|response|> tokens)
- stage2: Full model with reasoning (if applicable)

For stage1_inst_SFT:
- Prompts are automatically formatted as: <|instruction|> {question} <|response|>
- Generation stops at <|end|> token
- Response extraction removes the prompt and special tokens automatically
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import json
from typing import List, Dict, Any
from collections import Counter

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nemo.nemo_wrapper import create_modular_model_nemo
try:
    from src.nemo.config_loader import create_nemo_config_from_existing, ConfigLoader
except ImportError:
    create_nemo_config_from_existing = None
    ConfigLoader = None
from transformers import AutoTokenizer

def format_prompt_for_stage(prompt: str, stage: str, tokenizer=None) -> str:
    """
    Format prompt according to the training stage requirements.
    
    Args:
        prompt: Raw prompt string
        stage: Inference stage (stage1, stage1_inst_SFT, stage2)
        tokenizer: Optional tokenizer to check if tokens exist in vocab
    
    Returns:
        Formatted prompt string
    """
    if stage == "stage1_inst_SFT":
        # For stage1_inst_SFT, format should be: <|instruction|> {question} <|response|>
        instruction_token = "<|instruction|>"
        response_token = "<|response|>"
        
        # Check if prompt already has the format
        if instruction_token in prompt and response_token in prompt:
            # Already formatted, return as-is
            return prompt
        elif response_token in prompt:
            # Has response token but not instruction, add instruction at start
            return f"{instruction_token} {prompt}"
        elif instruction_token in prompt:
            # Has instruction but not response, add response at end
            return f"{prompt} {response_token}"
        else:
            # No special tokens, wrap the prompt
            return f"{instruction_token} {prompt} {response_token}"
    else:
        # For stage1 and stage2, return prompt as-is
        return prompt

def load_prompts_from_file(prompts_file: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Try common keys: 'prompt', 'text', 'input', 'question'
                prompt = item.get('prompt') or item.get('text') or item.get('input') or item.get('question')
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

def batch_tokenize_prompts(tokenizer, prompts: List[str], device, max_length: int = None, stage: str = "stage1") -> tuple:
    """
    Tokenize a batch of prompts with padding.
    
    Args:
        tokenizer: Tokenizer to use
        prompts: List of prompt strings
        device: Device to move tensors to
        max_length: Maximum length for padding (None = use longest sequence)
        stage: Inference stage (for formatting prompts)
    
    Returns:
        Tuple of (input_ids, attention_mask) tensors with shape (batch_size, seq_len)
    """
    # Format prompts according to stage
    formatted_prompts = [format_prompt_for_stage(prompt, stage, tokenizer) for prompt in prompts]
    
    # CRITICAL: Ensure pad_token_id is set and different from eos_token_id
    # This must be done BEFORE tokenization
    # The tokenizer's default pad_token_id (151643) decodes to <|endoftext|>, which is problematic
    # We need to use a token that doesn't decode to <|endoftext|> or <|im_end|>
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # Token 0 decodes to '!' which is safe for padding
    elif tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = 0  # Use 0 if pad_token_id equals eos_token_id
    else:
        # Verify pad_token_id doesn't decode to <|endoftext|>
        pad_decoded = tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False)
        if '<|endoftext|>' in pad_decoded or pad_decoded == tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False):
            tokenizer.pad_token_id = 0  # Force to 0 if it decodes to endoftext
    
    # Tokenize all prompts
    # For stage1_inst_SFT, don't add automatic special tokens (BOS/EOS) since we manually add instruction tokens
    add_special_tokens = stage != "stage1_inst_SFT"
    tokenized = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True if max_length else False,
        max_length=max_length,
        return_attention_mask=True,
        add_special_tokens=add_special_tokens,
        pad_to_multiple_of=None  # Don't pad to multiple
    )
    
    input_ids = tokenized["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = tokenized["attention_mask"].to(device=device, dtype=torch.long)
    
    return input_ids, attention_mask

def get_eos_token_id(tokenizer, stage: str) -> int:
    """
    Get the appropriate EOS token ID for the given stage.
    
    Args:
        tokenizer: Tokenizer to use
        stage: Training stage
    
    Returns:
        EOS token ID
    """
    if stage == "stage1_inst_SFT":
        # For stage1_inst_SFT, use <|end|> token as EOS
        end_token = "<|end|>"
        if hasattr(tokenizer, 'encode'):
            end_token_id = tokenizer.encode(end_token, add_special_tokens=False)
            if end_token_id:
                return end_token_id[0] if isinstance(end_token_id, list) else end_token_id
        # Fallback to default eos_token_id
        return tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    else:
        # For other stages, use default eos_token_id
        return tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

def extract_response_text(generated_text: str, prompt_text: str, stage: str) -> str:
    """
    Extract the response text from generated text, removing the prompt.
    
    Args:
        generated_text: Full generated text including prompt
        prompt_text: Original prompt text
        stage: Training stage
    
    Returns:
        Extracted response text
    """
    if stage == "stage1_inst_SFT":
        # For stage1_inst_SFT, remove everything up to and including <|response|>
        response_token = "<|response|>"
        if response_token in generated_text:
            # Find the position after <|response|>
            response_idx = generated_text.find(response_token)
            if response_idx != -1:
                # Extract text after <|response|> (including the token itself if it's part of generation)
                response_start = response_idx + len(response_token)
                response = generated_text[response_start:].strip()
                
                # Remove <|end|> token if present (and everything after it)
                end_token = "<|end|>"
                if end_token in response:
                    end_idx = response.find(end_token)
                    response = response[:end_idx].strip()
                
                # Also remove any <|endoftext|> tokens and everything after first occurrence
                endoftext_token = "<|endoftext|>"
                if endoftext_token in response:
                    endoftext_idx = response.find(endoftext_token)
                    response = response[:endoftext_idx].strip()
                
                # Clean up: remove any leading/trailing special tokens
                response = response.strip()
                
                return response
        # Fallback: if no response token found, try standard extraction
        if generated_text.startswith(prompt_text):
            response = generated_text[len(prompt_text):].strip()
            # Remove <|end|> and <|endoftext|> if present
            for token in ["<|end|>", "<|endoftext|>"]:
                if token in response:
                    response = response[:response.find(token)].strip()
            return response
        return generated_text
    else:
        # For other stages, standard extraction
        if generated_text.startswith(prompt_text):
            return generated_text[len(prompt_text):].strip()
        return generated_text

def run_batched_inference(
    model, 
    tokenizer, 
    prompts: List[str], 
    args, 
    device, 
    use_mixed_precision, 
    use_kv_cache,
    batch_size: int
) -> List[str]:
    """
    Run inference on a batch of prompts.
    
    Args:
        model: Model to use for inference
        tokenizer: Tokenizer to use
        prompts: List of prompt strings
        args: Arguments object
        device: Device to use
        use_mixed_precision: Whether to use mixed precision
        use_kv_cache: Whether to use KV cache
        batch_size: Batch size for processing
    
    Returns:
        List of generated text strings
    """
    if len(prompts) == 0:
        return []
    
    # Get the actual model that has generate() method
    actual_model = None
    has_generate = False
    
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
    
    # Use batched generation if available
    if use_kv_cache and has_generate:
        # Use DecoderOnlyModel.generate() from optimized_models.py
        # This method has full support for all parameters including repetition_penalty
        # and uses HuggingFace's optimized generation internally
        
        # Format prompts according to stage
        formatted_prompts = [format_prompt_for_stage(prompt, args.stage, tokenizer) for prompt in prompts]
        
        # Debug: Print first formatted prompt to verify format
        if args.stage == "stage1_inst_SFT" and len(formatted_prompts) > 0:
            print(f"🔍 First formatted prompt (sample): {formatted_prompts[0][:200]}...")
            # Verify it has the right tokens
            if "<|instruction|>" in formatted_prompts[0] and "<|response|>" in formatted_prompts[0]:
                print(f"✅ Prompt format is correct (has <|instruction|> and <|response|>)")
            else:
                print(f"⚠️  WARNING: Prompt format might be incorrect!")
                print(f"   Has <|instruction|>: {'<|instruction|>' in formatted_prompts[0]}")
                print(f"   Has <|response|>: {'<|response|>' in formatted_prompts[0]}")
        
        # Tokenize all prompts with padding
        input_ids, attention_mask = batch_tokenize_prompts(tokenizer, formatted_prompts, device, stage=args.stage)
        
        # Debug: Decode first input to verify tokenization
        if args.stage == "stage1_inst_SFT" and len(input_ids) > 0:
            first_input_ids = input_ids[0]
            first_decoded = tokenizer.decode(first_input_ids, skip_special_tokens=False)
            print(f"🔍 First tokenized prompt (decoded): {first_decoded[:200]}...")
            
            # Debug: Check what token IDs are in the padded positions
            pad_token_id = tokenizer.pad_token_id
            eos_token_id = tokenizer.eos_token_id
            print(f"🔍 pad_token_id: {pad_token_id}, eos_token_id: {eos_token_id}")
            
            # Find where the actual prompt starts (first non-pad token)
            if pad_token_id is not None:
                first_non_pad_idx = (first_input_ids != pad_token_id).nonzero(as_tuple=True)[0]
                if len(first_non_pad_idx) > 0:
                    actual_start_idx = first_non_pad_idx[0].item()
                    print(f"🔍 Actual prompt starts at index: {actual_start_idx}")
                    # Check what tokens are before the prompt
                    if actual_start_idx > 0:
                        padding_tokens = first_input_ids[:actual_start_idx].tolist()
                        padding_decoded = tokenizer.decode(padding_tokens, skip_special_tokens=False)
                        print(f"🔍 Padding tokens (first {min(10, len(padding_tokens))}): {padding_tokens[:10]}")
                        print(f"🔍 Padding decoded: {repr(padding_decoded[:100])}")
                        # Check if padding is using eos_token_id
                        if eos_token_id is not None and eos_token_id in padding_tokens:
                            print(f"⚠️  WARNING: Padding contains eos_token_id ({eos_token_id})! This is the problem!")
                            print(f"   pad_token_id should be {pad_token_id}, but padding is using {eos_token_id}")
            
            # Check if it has the right tokens
            if "<|instruction|>" in first_decoded and "<|response|>" in first_decoded:
                print(f"✅ Tokenized prompt format is correct")
            else:
                print(f"⚠️  WARNING: Tokenized prompt format might be incorrect!")
        
        # Get appropriate EOS token for the stage
        eos_token_id = get_eos_token_id(tokenizer, args.stage)
        
        # Debug: Print EOS token info and verify it's correct
        if args.stage == "stage1_inst_SFT":
            end_token = "<|end|>"
            # Verify the tokenizer can encode/decode it correctly
            end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
            if end_token_ids:
                expected_eos_id = end_token_ids[0] if isinstance(end_token_ids, list) else end_token_ids
                if eos_token_id != expected_eos_id:
                    print(f"⚠️  WARNING: EOS token ID mismatch! Expected {expected_eos_id}, got {eos_token_id}")
                    eos_token_id = expected_eos_id  # Fix it
            end_token_decoded = tokenizer.decode([eos_token_id], skip_special_tokens=False)
            print(f"🔍 EOS token ID: {eos_token_id}, decoded: '{end_token_decoded}'")
            print(f"🔍 Expected token: '{end_token}'")
            if end_token_decoded != end_token:
                print(f"⚠️  WARNING: EOS token mismatch! Expected '{end_token}', got '{end_token_decoded}'")
                # Try to fix it
                end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
                if end_token_ids:
                    eos_token_id = end_token_ids[0] if isinstance(end_token_ids, list) else end_token_ids
                    print(f"🔧 Fixed EOS token ID to: {eos_token_id}")
        
        # Prepare generation kwargs
        do_sample = not args.greedy
        temperature = 1.0 if args.greedy else args.temperature
        top_k = 1 if args.greedy else (args.top_k if args.top_k > 0 else 50)
        
        # Set pad_token_id - make sure it's different from eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id == eos_token_id:
            # If pad and eos are the same, use unk_token_id or 0
            pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
        
        # CRITICAL: For left-padded sequences, HuggingFace generate() needs special handling
        # The attention_mask should correctly mask padding, but we need to ensure
        # that generation starts from the actual prompt, not from padding tokens
        # HuggingFace's generate() should handle this automatically with attention_mask,
        # but let's verify the input_ids and attention_mask are correct
        
        # Debug: Check first example's input_ids and attention_mask
        if args.stage == "stage1_inst_SFT" and len(input_ids) > 0:
            first_input_ids = input_ids[0].cpu().tolist()
            first_attention_mask = attention_mask[0].cpu().tolist()
            # Find where actual prompt starts (first non-padding token)
            if pad_token_id is not None:
                non_pad_indices = [i for i, token_id in enumerate(first_input_ids) if token_id != pad_token_id]
                if non_pad_indices:
                    prompt_start_idx = non_pad_indices[0]
                    print(f"🔍 First example - Prompt starts at index {prompt_start_idx}")
                    print(f"   Input length: {len(first_input_ids)}, Attention mask sum: {sum(first_attention_mask)}")
                    # Decode the actual prompt part (without padding)
                    actual_prompt_ids = first_input_ids[prompt_start_idx:]
                    actual_prompt_text = tokenizer.decode(actual_prompt_ids, skip_special_tokens=False)
                    print(f"   Actual prompt (decoded): {actual_prompt_text[:150]}...")
        
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,  # CRITICAL: This masks out padding tokens
            'max_new_tokens': args.max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': 0.95,
            'do_sample': do_sample,
            'repetition_penalty': args.repetition_penalty,
            'eos_token_id': eos_token_id,
            'pad_token_id': pad_token_id
        }
        
        print(f"🔍 Generation kwargs - eos_token_id: {eos_token_id}, pad_token_id: {pad_token_id}, max_new_tokens: {args.max_tokens}")
        
        # Also update the model's generation_config if it exists
        if hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'generation_config'):
            actual_model.decoder.generation_config.eos_token_id = eos_token_id
            actual_model.decoder.generation_config.pad_token_id = pad_token_id
            print(f"🔧 Updated decoder.generation_config - eos_token_id: {eos_token_id}, pad_token_id: {pad_token_id}")
        
        # Generate for all prompts in batch
        # Choose between HuggingFace's generate() or custom generate()
        use_custom = getattr(args, 'use_custom_generate', False)
        
        if use_custom:
            # Use custom (non-HuggingFace) generate function
            print("🔧 Using custom (non-HuggingFace) generate function")
            if hasattr(actual_model, 'generate_custom'):
                with torch.no_grad():
                    if use_mixed_precision:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            generated_ids = actual_model.generate_custom(**generate_kwargs)
                    else:
                        generated_ids = actual_model.generate_custom(**generate_kwargs)
            else:
                print("⚠️  WARNING: Custom generate not available, falling back to HuggingFace generate")
                use_custom = False
        
        if not use_custom:
            # Use HuggingFace's generate() which automatically:
            # 1. Use attention_mask to ignore padding tokens
            # 2. Start generation from the last non-padding token
            # 3. Return full sequence (input_ids + generated tokens)
            print("🔧 Using HuggingFace's generate function")
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        generated_ids = actual_model.generate(**generate_kwargs)
                else:
                    generated_ids = actual_model.generate(**generate_kwargs)
        
        # Decode all generated texts
        # CRITICAL: HuggingFace's generate() returns the FULL sequence (input_ids + newly generated tokens)
        # We need to extract only the newly generated tokens
        generated_texts = []
        for i, prompt in enumerate(prompts):
            # Get the input and generated sequences
            input_seq = input_ids[i]
            generated_seq = generated_ids[i]
            
            # Get lengths
            input_len = input_seq.shape[0] if isinstance(input_seq, torch.Tensor) else len(input_seq)
            generated_len = generated_seq.shape[0] if isinstance(generated_seq, torch.Tensor) else len(generated_seq)
            
            # Extract only the newly generated tokens (after the full input sequence)
            # HuggingFace's generate() returns: [input_ids] + [new_tokens]
            # So we extract everything after input_len
            if generated_len > input_len:
                new_tokens = generated_seq[input_len:] if isinstance(generated_seq, torch.Tensor) else generated_seq[input_len:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            else:
                generated_text = ""
            
            # Decode the full input sequence (including padding) for prompt text
            # This is needed for extract_response_text to work correctly
            prompt_text = tokenizer.decode(input_seq, skip_special_tokens=False)
            
            # Debug: Print first example to see what's being generated
            if i == 0:
                print(f"\n🔍 DEBUG: First example generation ({args.stage}):")
                print(f"   Input length: {input_len}, Generated length: {generated_len}")
                print(f"   New tokens: {generated_len - input_len if generated_len > input_len else 0}")
                print(f"   Prompt text: {prompt_text[:150]}...")
                print(f"   Generated text (new tokens only): {generated_text[:200]}...")
                if args.stage == "stage1_inst_SFT":
                    print(f"   Has <|response|> in prompt: {'<|response|>' in prompt_text}")
                    print(f"   Has <|end|> in generated: {'<|end|>' in generated_text}")
            
            # Extract response text using stage-specific extraction
            # Combine prompt + generated for extraction (extract_response_text expects full sequence)
            full_generated_text = prompt_text + generated_text
            new_text = extract_response_text(full_generated_text, prompt_text, args.stage)
            
            generated_texts.append(new_text)
        
        return generated_texts
    else:
        # Fallback to sequential processing if batching not supported
        generated_texts = []
        for prompt in prompts:
            generated_text = run_single_inference(model, tokenizer, prompt, args, device, use_mixed_precision, use_kv_cache)
            generated_texts.append(generated_text)
        return generated_texts

def run_single_inference(model, tokenizer, prompt: str, args, device, use_mixed_precision, use_kv_cache) -> str:
    """Run inference on a single prompt and return the generated text."""
    # Format prompt according to stage
    formatted_prompt = format_prompt_for_stage(prompt, args.stage, tokenizer)
    
    # Prepare input
    # For stage1_inst_SFT, don't add automatic special tokens (BOS/EOS) since we manually add instruction tokens
    add_special_tokens = args.stage != "stage1_inst_SFT"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
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
        
        # Get appropriate EOS token for the stage
        eos_token_id = get_eos_token_id(tokenizer, args.stage)
        
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
            'eos_token_id': eos_token_id,
            'pad_token_id': tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        }
        
        # Choose between HuggingFace's generate() or custom generate()
        use_custom = getattr(args, 'use_custom_generate', False)
        
        with torch.no_grad():
            if use_custom and hasattr(actual_model, 'generate_custom'):
                # Use custom (non-HuggingFace) generate function
                print("🔧 Using custom (non-HuggingFace) generate function")
                if use_mixed_precision:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        generated_ids = actual_model.generate_custom(**generate_kwargs)
                else:
                    generated_ids = actual_model.generate_custom(**generate_kwargs)
            else:
                # Use HuggingFace's generate()
                if use_custom:
                    print("⚠️  WARNING: Custom generate not available, falling back to HuggingFace generate")
                else:
                    print("🔧 Using HuggingFace's generate function")
                if use_mixed_precision:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        generated_ids = actual_model.generate(**generate_kwargs)
                else:
                    generated_ids = actual_model.generate(**generate_kwargs)
        
        # Decode the generated text (keep special tokens for stage1_inst_SFT)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # Extract response text using stage-specific extraction
        new_text = extract_response_text(generated_text, prompt_text, args.stage)
        
        return new_text
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
                
                # Apply repetition penalty (frequency-based for stronger penalty on repeated tokens)
                if args.repetition_penalty != 1.0 and len(generated_tokens) > 0:
                    # Count frequency of each token
                    token_counts = Counter(generated_tokens)
                    
                    # Apply penalty based on frequency (more repetitions = stronger penalty)
                    for token_id, count in token_counts.items():
                        # Apply penalty multiple times based on count (log scale to avoid extreme values)
                        penalty_factor = args.repetition_penalty ** min(count, 5)  # Cap at 5x to avoid extreme values
                        if last_token_logits[token_id] < 0:
                            last_token_logits[token_id] *= penalty_factor
                        else:
                            last_token_logits[token_id] /= penalty_factor
                
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
                
                # Stop if we hit a special token (use stage-specific EOS token)
                eos_token_id = get_eos_token_id(tokenizer, args.stage)
                if next_token_id.item() == eos_token_id:
                    break
        
        # Decode generated tokens (keep special tokens for stage1_inst_SFT)
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        # Extract response text using stage-specific extraction
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        full_generated = prompt_text + generated_text
        return extract_response_text(full_generated, prompt_text, args.stage)

def main():
    parser = argparse.ArgumentParser(description="NeMo ModularModel Text Generation")
    parser.add_argument("--model_config", type=str, default="model_config_243M",
                       help="Model configuration key (e.g., model_config_243M, model_config_1.8B)")
    parser.add_argument("--stage", type=str, default="stage1",
                       help="Inference stage (stage1, stage1_inst_SFT, stage2)")
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
    parser.add_argument("--use_custom_generate", action="store_true", default=False,
                       help="Use custom (non-HuggingFace) generate function instead of HuggingFace's generate")
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
                       help="Batch size for batched inference when processing multiple prompts (default: 1, use >1 for faster processing)")
    
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
    
    if args.prompts_file:
        print(f"📊 Batch size: {args.batch_size} (batched inference enabled)")
        if args.batch_size > 1 and use_kv_cache:
            print(f"💡 Using batched inference for faster processing")
        elif args.batch_size > 1 and not use_kv_cache:
            print(f"⚠️ Batch size > 1 but KV cache disabled - will fall back to sequential processing")
    
    # Load configuration using ConfigLoader.create_nemo_model_config to ensure config matches the stage
    if ConfigLoader is not None:
        config_loader = ConfigLoader()
        config = config_loader.create_nemo_model_config(args.model_config, args.stage)
        print(f"📊 Loaded configuration using ConfigLoader.create_nemo_model_config for stage: {args.stage}")
    elif create_nemo_config_from_existing is not None:
        # Fallback to convenience function if ConfigLoader is not available
        config = create_nemo_config_from_existing(args.model_config, args.stage)
        print(f"📊 Loaded configuration using create_nemo_config_from_existing for stage: {args.stage}")
    else:
        raise RuntimeError("Neither ConfigLoader nor create_nemo_config_from_existing is available")
    
    # Echo weight tying resolved from config for transparency
    if 'tie_weights' in config:
        print(f"📊 Weight tying (from config): {'Enabled' if config['tie_weights'] else 'Disabled'}")
    else:
        print("📊 Weight tying: not specified in config (will use model default)")
    
    # Show model architecture stage from config
    print(f"📊 Model architecture stage from config: {config.get('training_stage', 'not set')}")
    
    # Set mixed precision based on arguments and device capability
    config['mixed_precision'] = "bf16" if use_mixed_precision else None
    # Keep flash attention enabled to match training (unless explicitly disabled)
    # Flash attention should produce same results as standard attention, just faster
    if not hasattr(args, 'disable_flash_attention') or not args.disable_flash_attention:
        config['use_flash_attention'] = True  # Match training configuration
    
    # ============================================================================
    # TOKENIZER SETUP - SEPARATE PATHS FOR stage1 AND stage1_inst_SFT
    # ============================================================================
    # This matches the training workflow exactly:
    # - stage1: Plain tokenizer, no special instruction tokens
    # - stage1_inst_SFT: Tokenizer with instruction format special tokens
    # ============================================================================
    
    tokenizer_path = config.get("tokenizer_path", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    new_tokens_added = []
    original_vocab_size = None
    actual_vocab_size = None
    
    if args.stage == "stage1_inst_SFT":
        # ========================================================================
        # STAGE1_INST_SFT: Instruction Fine-Tuning Setup
        # Matches: ModularModelstage1_InstructionSFT.py train_production_mode
        # ========================================================================
        print("=" * 80)
        print("🔤 STAGE1_INST_SFT: Loading tokenizer with instruction format special tokens...")
        print("=" * 80)
        
        # Load tokenizer with caching support (same as training)
        from src.utils.tokenizer_manager import get_tokenizer_with_caching
        tokenizer = get_tokenizer_with_caching(
            tokenizer_path=tokenizer_path,
            custom_tokens=None,  # Use default special tokens
            force_download=False,
            cache_dir="tokenizers"
        )
        print(f"✅ Loaded tokenizer from {tokenizer_path}")
        
        # CRITICAL: Set padding_side to 'left' for decoder-only models
        # This ensures correct generation behavior (right-padding causes issues)
        tokenizer.padding_side = 'left'
        print(f"🔧 Set tokenizer padding_side to 'left' for decoder-only generation")
        
        # Add instruction format special tokens if not already present (same as training)
        instruction_token = "<|instruction|>"
        response_token = "<|response|>"
        end_token = "<|end|>"
        
        instruction_tokens = [instruction_token, response_token, end_token]
        new_tokens = []
        for token in instruction_tokens:
            if token not in tokenizer.get_vocab():
                new_tokens.append(token)
        
        # Store original vocab size before adding tokens (for model creation)
        original_vocab_size = config.get("vocab_size")
        
        if new_tokens:
            print(f"➕ Adding {len(new_tokens)} instruction format tokens: {new_tokens}")
            tokenizer.add_tokens(new_tokens)
            new_tokens_added = new_tokens
            actual_vocab_size = len(tokenizer)
            print(f"✅ Tokenizer vocab size after adding instruction tokens: {actual_vocab_size}")
            print(f"   Original vocab size: {original_vocab_size}")
            print(f"   New vocab size: {actual_vocab_size}")
            print(f"   Added {actual_vocab_size - original_vocab_size} tokens")
        else:
            print(f"✅ All instruction format tokens already present in tokenizer")
            actual_vocab_size = len(tokenizer)
            print(f"   Tokenizer vocab size: {actual_vocab_size}")
        
        # CRITICAL: Set pad_token_id to a token that's NOT eos_token_id and doesn't decode to <|endoftext|>
        # When padding on the left, we don't want to use eos_token_id as padding
        # This prevents the model from seeing <|endoftext|> tokens at the start
        # The tokenizer's default pad_token_id (151643) decodes to <|endoftext|>, which is problematic
        pad_decoded = tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False) if tokenizer.pad_token_id is not None else None
        eos_decoded = tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False) if tokenizer.eos_token_id is not None else None
        
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id or (pad_decoded and '<|endoftext|>' in pad_decoded):
            # Use token 0 which decodes to '!' - safe for padding
            old_pad_token_id = tokenizer.pad_token_id
            tokenizer.pad_token_id = 0
            print(f"🔧 Set pad_token_id to 0 (was: {old_pad_token_id})")
            print(f"   Reason: pad_token_id decoded to '{pad_decoded}' which contains <|endoftext|>")
            print(f"   Token 0 decodes to: '{tokenizer.decode([0], skip_special_tokens=False)}'")
        else:
            print(f"✅ pad_token_id already set correctly: {tokenizer.pad_token_id}")
            print(f"   pad_token_id decodes to: '{pad_decoded}'")
        
        print("=" * 80)
        
    elif args.stage == "stage1":
        # ========================================================================
        # STAGE1: Next Token Prediction Setup (Plain Tokenizer)
        # Matches: ModularModelstage1_NTPtraining.py train_production_mode
        # ========================================================================
        print("=" * 80)
        print("🔤 STAGE1: Loading tokenizer (no special instruction tokens)...")
        print("=" * 80)
        
        # Load tokenizer with caching support (same as training)
        from src.utils.tokenizer_manager import get_tokenizer_with_caching
        tokenizer = get_tokenizer_with_caching(
            tokenizer_path=tokenizer_path,
            custom_tokens=None,  # Use default special tokens
            force_download=False,
            cache_dir="tokenizers"
        )
        print(f"✅ Loaded tokenizer from {tokenizer_path}")
        
        # CRITICAL: Set padding_side to 'left' for decoder-only models
        tokenizer.padding_side = 'left'
        print(f"🔧 Set tokenizer padding_side to 'left' for decoder-only generation")
        
        # No special tokens for stage1 - plain tokenizer
        original_vocab_size = len(tokenizer)
        actual_vocab_size = original_vocab_size
        print(f"✅ Tokenizer vocab size: {actual_vocab_size}")
        print("=" * 80)
        
    else:
        # ========================================================================
        # OTHER STAGES: Default tokenizer loading
        # ========================================================================
        print(f"🔤 Loading tokenizer for stage: {args.stage}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ Loaded tokenizer from {tokenizer_path}")
        original_vocab_size = len(tokenizer)
        actual_vocab_size = original_vocab_size
        print(f"✅ Tokenizer vocab size: {actual_vocab_size}")
    
    # ============================================================================
    # CHECKPOINT VOCAB SIZE CHECK (for stage1_inst_SFT)
    # ============================================================================
    # For stage1_inst_SFT, we need to check checkpoint vocab size to determine
    # if we need to resize the model after loading (same as training workflow)
    checkpoint_vocab_size = None
    if args.stage == "stage1_inst_SFT" and os.path.exists(args.checkpoint):
        print(f"🔍 Checking checkpoint vocab size for stage1_inst_SFT...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            # Look for embedding weights in checkpoint to determine vocab size
            for key, value in checkpoint['state_dict'].items():
                # Check for embed_tokens.weight (with or without model. prefix)
                if 'embed_tokens.weight' in key:
                    checkpoint_vocab_size = value.shape[0]
                    print(f"📊 Checkpoint vocab size (from embed_tokens): {checkpoint_vocab_size}")
                    break
                # Also check for lm_head.weight as fallback
                elif 'lm_head.weight' in key and checkpoint_vocab_size is None:
                    checkpoint_vocab_size = value.shape[0]
                    print(f"📊 Checkpoint vocab size (from lm_head): {checkpoint_vocab_size}")
        
        if checkpoint_vocab_size is None:
            print(f"⚠️ Could not determine checkpoint vocab size, using original vocab size: {original_vocab_size}")
            checkpoint_vocab_size = original_vocab_size
        else:
            print(f"✅ Checkpoint vocab size: {checkpoint_vocab_size}")
            print(f"   Original vocab size (before adding tokens): {original_vocab_size}")
            print(f"   Actual vocab size (after adding tokens): {actual_vocab_size}")
            print(f"   Expected checkpoint vocab size (should match actual): {actual_vocab_size}")
            if checkpoint_vocab_size == actual_vocab_size:
                print(f"   ✅ Checkpoint vocab size matches tokenizer vocab size (includes instruction tokens)")
            elif checkpoint_vocab_size == original_vocab_size:
                print(f"   ⚠️  WARNING: Checkpoint vocab size matches ORIGINAL vocab size (missing instruction tokens!)")
                print(f"      This checkpoint might not have been trained with instruction tokens!")
            else:
                print(f"   ⚠️  WARNING: Checkpoint vocab size ({checkpoint_vocab_size}) doesn't match expected sizes!")
                print(f"      Expected either {original_vocab_size} (original) or {actual_vocab_size} (with tokens)")
    
    # ============================================================================
    # MODEL CREATION - SEPARATE PATHS FOR stage1 AND stage1_inst_SFT
    # ============================================================================
    
    if args.stage == "stage1_inst_SFT":
        # ========================================================================
        # STAGE1_INST_SFT: Create model with ORIGINAL vocab size, then resize to 
        # checkpoint vocab size BEFORE loading checkpoint (inference-specific workflow)
        # ========================================================================
        print("=" * 80)
        print(f"📦 STAGE1_INST_SFT: Creating model with original vocab size ({original_vocab_size})")
        if checkpoint_vocab_size is not None and checkpoint_vocab_size != original_vocab_size:
            print(f"   Will resize to checkpoint vocab size ({checkpoint_vocab_size}) BEFORE loading checkpoint")
        print("=" * 80)
        
        # IMPORTANT: For inference, create model with ORIGINAL vocab size first
        # Then resize to checkpoint vocab size BEFORE loading (different from training)
        config["vocab_size"] = original_vocab_size
        
        # Create model - use stage1 for model architecture (decoder-only)
        # The inference stage is stage1_inst_SFT, but model architecture is decoder-only (stage1)
        config["training_stage"] = "stage1"  # Model is decoder-only (stage1), not full modular model
        print(f"📊 Setting model architecture to 'stage1' (decoder-only) for inference stage 'stage1_inst_SFT'")
        
    elif args.stage == "stage1":
        # ========================================================================
        # STAGE1: Create model with tokenizer vocab size (no resizing needed)
        # ========================================================================
        print("=" * 80)
        print(f"📦 STAGE1: Creating model with vocab size ({actual_vocab_size})")
        print("=" * 80)
        
        config["vocab_size"] = actual_vocab_size
        config["training_stage"] = "stage1"  # Decoder-only model
        print(f"📊 Setting model architecture to 'stage1' (decoder-only)")
        
    else:
        # ========================================================================
        # OTHER STAGES: Use config vocab size
        # ========================================================================
        print(f"📦 Creating model for stage: {args.stage}")
        if "vocab_size" not in config:
            config["vocab_size"] = actual_vocab_size
    
    # Create model
    print(f"📊 Creating model with architecture stage: {config.get('training_stage')}, vocab_size: {config.get('vocab_size')}")
    model = create_modular_model_nemo(**config)
    
    # ============================================================================
    # RESIZE MODEL FOR INFERENCE (stage1_inst_SFT only)
    # ============================================================================
    # For inference: Resize model to match checkpoint vocab size BEFORE loading
    # This is different from training which resizes AFTER loading
    if args.stage == "stage1_inst_SFT" and checkpoint_vocab_size is not None:
        if checkpoint_vocab_size != original_vocab_size:
            print("=" * 80)
            print(f"🔧 INFERENCE: Resizing model to match checkpoint vocab size BEFORE loading...")
            print(f"   Model vocab size: {original_vocab_size} → Checkpoint vocab size: {checkpoint_vocab_size}")
            print("=" * 80)
            
            # Import resize function from training module
            try:
                from src.nemo.ModularModelstage1_InstructionSFT import resize_model_embeddings
                import logging
                
                # Resize model embeddings to match checkpoint vocab size
                tie_weights_config = config.get("tie_weights", False)
                resize_success = resize_model_embeddings(
                    model, 
                    checkpoint_vocab_size, 
                    logging, 
                    tie_weights=tie_weights_config
                )
                
                if resize_success:
                    print(f"✅ Model resized successfully to vocab size: {checkpoint_vocab_size}")
                else:
                    print(f"⚠️ Model resize failed - continuing anyway (may cause issues)")
            except Exception as e:
                print(f"⚠️ Could not resize model embeddings: {e}")
                print(f"⚠️ This may cause issues when loading checkpoint")
                import traceback
                traceback.print_exc()
    
    # Load checkpoint with proper dtype handling
    if os.path.exists(args.checkpoint):
        # PyTorch 2.6+ defaults to weights_only=True for security, but checkpoints may contain
        # tokenizer objects (e.g., Qwen2TokenizerFast) which require weights_only=False
        # Note: We already loaded the checkpoint above to check vocab size, but we need to reload it
        # to avoid any potential issues with the checkpoint object being modified
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
            
            # Verify vocab size matches tokenizer (especially important for stage1_inst_SFT)
            tokenizer_vocab_size = len(tokenizer)
            model_vocab_size = None
            
            # Try to get model vocab size from embedding layer
            try:
                if hasattr(model, 'modular_model') and hasattr(model.modular_model, 'model'):
                    if hasattr(model.modular_model.model, 'decoder') and hasattr(model.modular_model.model.decoder, 'embed_tokens'):
                        model_vocab_size = model.modular_model.model.decoder.embed_tokens.weight.shape[0]
                    elif hasattr(model.modular_model.model, 'decoder') and hasattr(model.modular_model.model.decoder, 'vocab_size'):
                        model_vocab_size = model.modular_model.model.decoder.vocab_size
                elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                    if hasattr(model.model.decoder, 'embed_tokens'):
                        model_vocab_size = model.model.decoder.embed_tokens.weight.shape[0]
                    elif hasattr(model.model.decoder, 'vocab_size'):
                        model_vocab_size = model.model.decoder.vocab_size
            except Exception as e:
                print(f"⚠️ Could not determine model vocab size: {e}")
            
            if model_vocab_size is not None:
                print(f"📊 Model vocab size: {model_vocab_size}, Tokenizer vocab size: {tokenizer_vocab_size}")
                if model_vocab_size != tokenizer_vocab_size:
                    print(f"⚠️ WARNING: Model vocab size ({model_vocab_size}) doesn't match tokenizer vocab size ({tokenizer_vocab_size})")
                    if args.stage == "stage1_inst_SFT":
                        print(f"   Note: For inference, model was resized to match checkpoint vocab size before loading")
                        print(f"   If sizes still don't match, there may be an issue with the checkpoint or tokenizer")
                else:
                    print(f"✅ Model and tokenizer vocab sizes match: {model_vocab_size}")
            
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
        print(f"📦 Batch size: {args.batch_size}")
        print(f"🔄 Running inference on {len(prompts)} prompts...\n")
        
        results = []
        
        # Process prompts in batches
        num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            
            print(f"\n📦 Batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} prompts)")
            print(f"   Processing prompts {start_idx + 1}-{end_idx}...")
            
            try:
                # Run batched inference
                batch_generated_texts = run_batched_inference(
                    model, 
                    tokenizer, 
                    batch_prompts, 
                    args, 
                    device, 
                    use_mixed_precision, 
                    use_kv_cache,
                    args.batch_size
                )
                
                # Add results for this batch
                for i, (prompt, generated_text) in enumerate(zip(batch_prompts, batch_generated_texts)):
                    prompt_idx = start_idx + i + 1
                    results.append({
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "prompt_index": prompt_idx
                    })
                    print(f"   [{prompt_idx}/{len(prompts)}] ✅ Generated {len(generated_text)} characters")
                
            except Exception as e:
                print(f"   ❌ Error processing batch {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to individual processing for this batch
                print(f"   🔄 Falling back to individual processing for this batch...")
                for i, prompt in enumerate(batch_prompts):
                    prompt_idx = start_idx + i + 1
                    try:
                        generated_text = run_single_inference(model, tokenizer, prompt, args, device, use_mixed_precision, use_kv_cache)
                        results.append({
                            "prompt": prompt,
                            "generated_text": generated_text,
                            "prompt_index": prompt_idx
                        })
                        print(f"   [{prompt_idx}/{len(prompts)}] ✅ Generated {len(generated_text)} characters")
                    except Exception as e2:
                        print(f"   [{prompt_idx}/{len(prompts)}] ❌ Error: {e2}")
                        results.append({
                            "prompt": prompt,
                            "generated_text": "",
                            "error": str(e2),
                            "prompt_index": prompt_idx
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
    
    # Format prompt according to stage
    formatted_prompt = format_prompt_for_stage(args.prompt, args.stage, tokenizer)
    if formatted_prompt != args.prompt:
        print(f"📝 Formatted prompt for {args.stage}: {formatted_prompt}")
    
    # Prepare input with matching dtype
    # For stage1_inst_SFT, don't add automatic special tokens (BOS/EOS) since we manually add instruction tokens
    add_special_tokens = args.stage != "stage1_inst_SFT"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
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
    print(f"📝 Formatted prompt: {formatted_prompt}")
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
            print("📊 Using DecoderOnlyModel.generate() from optimized_models.py")
        elif hasattr(model, 'model') and hasattr(model.model, 'generate'):
            actual_model = model.model
            print("📊 Using DecoderOnlyModel.generate() from optimized_models.py")
        elif hasattr(model, 'generate'):
            # Check if this is the wrapper - if so, we need to go deeper
            if hasattr(model, 'modular_model'):
                # This is ModularModelNeMoWrapper, go to the actual model
                if hasattr(model.modular_model, 'model'):
                    actual_model = model.modular_model.model
                    print("📊 Using DecoderOnlyModel.generate() from optimized_models.py")
                else:
                    actual_model = model.modular_model
                    print("📊 Using model.modular_model.generate()")
            else:
                actual_model = model
                print("📊 Using model.generate()")
        else:
            print("❌ Could not find generate() method, falling back to standard generation")
            raise AttributeError("No generate() method found")
        
        # Use DecoderOnlyModel.generate() from optimized_models.py
        # This method has full support for all parameters including repetition_penalty,
        # length_penalty, early_stopping, use_cache, etc.
        # It internally uses HuggingFace's optimized generation via decoder.generate()
        # CRITICAL: Use BF16 autocast to match training precision (not FP16)
        
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
        
        # Get appropriate EOS token for the stage
        eos_token_id = get_eos_token_id(tokenizer, args.stage)
        
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': args.max_tokens,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': 0.95,  # Default top_p for nucleus sampling
            'do_sample': do_sample,
            'repetition_penalty': args.repetition_penalty,
            'eos_token_id': eos_token_id,
            'pad_token_id': tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        }
        
        if use_mixed_precision:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated_ids = actual_model.generate(**generate_kwargs)
        else:
            generated_ids = actual_model.generate(**generate_kwargs)
        
        # Decode the generated text (keep special tokens for stage1_inst_SFT)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        
        # Extract response text using stage-specific extraction
        new_text = extract_response_text(generated_text, prompt_text, args.stage)
        
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
                
                # Apply repetition penalty (frequency-based for stronger penalty on repeated tokens)
                if args.repetition_penalty != 1.0 and len(generated_token_ids) > 0:
                    # Count frequency of each token
                    token_counts = Counter(generated_token_ids)
                    
                    # Apply penalty based on frequency (more repetitions = stronger penalty)
                    for token_id, count in token_counts.items():
                        # Apply penalty multiple times based on count (log scale to avoid extreme values)
                        penalty_factor = args.repetition_penalty ** min(count, 5)  # Cap at 5x to avoid extreme values
                        if last_token_logits[token_id] < 0:
                            last_token_logits[token_id] *= penalty_factor
                        else:
                            last_token_logits[token_id] /= penalty_factor
                
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
                
                # Stop if we hit a special token (use stage-specific EOS token)
                eos_token_id = get_eos_token_id(tokenizer, args.stage)
                if args.stage == "stage1_inst_SFT":
                    # For stage1_inst_SFT, check for <|end|> token
                    if next_token.strip() == "<|end|>" or next_token_id.item() == eos_token_id:
                        break
                else:
                    # For other stages, check for standard EOS tokens
                    if next_token in ['<|endoftext|>', '<|end|>', '\n\n'] or next_token_id.item() == eos_token_id:
                        break
            
            # Decode generated tokens (keep special tokens for stage1_inst_SFT)
            generated_text_str = ''.join(generated_tokens)
            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            full_generated = prompt_text + generated_text_str
            
            # Extract response text using stage-specific extraction
            extracted_response = extract_response_text(full_generated, prompt_text, args.stage)
            
            print(f"\n\n✅ Generation complete! Generated {len(generated_tokens)} tokens.")
            print(f"📊 Full generated text: {full_generated}")
            print(f"📊 Extracted response: {extracted_response}")
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

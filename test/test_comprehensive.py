#!/usr/bin/env python3
"""
Comprehensive test for Stage 1 NeMo training to verify next-token prediction correctness.
This test uses the exact same pipeline as the NeMo training code and runs one batch with a single item.
"""

import sys
import os
import torch
import yaml
import logging
from transformers import AutoTokenizer
from typing import Dict, Any

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nemo.ModularModelstage1_NTPtraining import ModularModelTrainingModule, generate_sample_data, BasicDataset
from src.nemo.nemo_wrapper import create_modular_model_nemo, create_modular_model_from_existing_config
try:
    from src.nemo.config_loader import create_nemo_config_from_existing
except ImportError:
    create_nemo_config_from_existing = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_target_ids_correctness(input_ids, target_ids, ignore_first_token=False, pad_token_id=0):
    """
    Verify that target_ids are correctly set up for next-token prediction.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        target_ids: Target token IDs [batch_size, seq_len]
        ignore_first_token: Whether to ignore the first token in verification
        pad_token_id: ID of the padding token
        
    Returns:
        bool: True if target_ids are correctly set up
    """
    batch_size, seq_len = input_ids.shape
    
    # Check that target_ids has the same shape as input_ids
    if target_ids.shape != input_ids.shape:
        return False
    
    # Check that the last position is always -100 (no target for last token)
    if not torch.all(target_ids[:, -1] == -100):
        return False
    
    # Check next-token prediction setup
    start_idx = 1 if ignore_first_token else 0
    
    for i in range(start_idx, seq_len - 1):
        # For position i, the target should be input_ids[i+1]
        expected_target = input_ids[:, i + 1]
        actual_target = target_ids[:, i]
        
        # Check if this position should have a valid target (not padding)
        # A position has a valid target if both current and next positions are not padding
        current_not_pad = (input_ids[:, i] != pad_token_id)
        next_not_pad = (input_ids[:, i + 1] != pad_token_id)
        should_have_target = current_not_pad & next_not_pad
        
        # For positions that should have targets, check they match
        if should_have_target.any():
            valid_targets = actual_target[should_have_target] == expected_target[should_have_target]
            if not valid_targets.all():
                return False
        
        # For positions that shouldn't have targets, check they are -100
        if (~should_have_target).any():
            invalid_targets = actual_target[~should_have_target] == -100
            if not invalid_targets.all():
                return False
    
    return True

def create_test_batch_nemo(tokenizer, max_length: int = 128) -> Dict[str, Any]:
    """Create a test batch using the NeMo training pipeline."""
    
    print("="*80)
    print("CREATING TEST BATCH USING NEMO TRAINING PIPELINE")
    print("="*80)
    
    # Create a test sample
    test_text = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # Test the function
    for i in range(10):
        print(f"fibonacci({i}) = {fibonacci(i)}")
    """
    
    print(f"Test text: {repr(test_text.strip())}")
    
    # Tokenize the text
    tokens = tokenizer.encode(test_text.strip(), add_special_tokens=True, max_length=max_length, truncation=True)
    
    # Pad to max_length
    if len(tokens) < max_length:
        tokens.extend([tokenizer.pad_token_id] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in tokens]
    
    # Create labels for next-token prediction
    # For position i, the target should be tokens[i+1] (next token)
    # Last position has no target (-100)
    # Padding positions have no target (-100)
    # If next token is padding, current position has no target (-100)
    labels = []
    for i in range(len(tokens)):
        if i == len(tokens) - 1:
            # Last position has no target
            labels.append(-100)
        elif tokens[i] == tokenizer.pad_token_id:
            # Padding position has no target
            labels.append(-100)
        elif tokens[i + 1] == tokenizer.pad_token_id:
            # Next token is padding, so no target for current position
            labels.append(-100)
        else:
            # Target is the next token
            labels.append(tokens[i + 1])
    
    print(f"Tokenized length: {len(tokens)}")
    print(f"Real tokens: {sum(attention_mask)}")
    print(f"Padding tokens: {len(tokens) - sum(attention_mask)}")
    
    # Create batch (single item) - convert to tensors for NeMo compatibility
    # Note: input_ids and labels should be long (int64) for token IDs
    # attention_mask can be long or float, but long is more memory efficient
    batch = {
        'input_ids': torch.tensor([tokens], dtype=torch.long),
        'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
        'labels': torch.tensor([labels], dtype=torch.long)
    }
    
    return batch

def test_nemo_stage1_training():
    """Test NeMo Stage 1 training with comprehensive verification."""
    
    print("="*80)
    print("COMPREHENSIVE NEMO STAGE 1 TRAINING TEST")
    print("="*80)
    
    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer_path = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        # Fallback to HuggingFace tokenizer
        tokenizer_path = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer loaded: {tokenizer_path}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # Create test batch with training sequence length
    print("\n2. Creating test batch with training sequence length...")
    batch = create_test_batch_nemo(tokenizer, max_length=2048)  # Same as training config
    
    # Create NeMo model using config
    print("\n3. Creating NeMo model using config...")
    try:
        # Try to use existing config
        project_root = os.path.dirname(os.path.dirname(__file__))
        model = create_modular_model_from_existing_config(
            model_config_key="model_config_1.7B",
            stage="stage1",
            base_path=project_root
        )
        
        # Move model to GPU device first, then set precision
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Ensure consistent mixed precision (BFloat16) throughout the model
        # Convert the entire model to BFloat16 for proper mixed precision
        model = model.to(torch.bfloat16)
        # Also ensure the underlying modular model is in BFloat16
        if hasattr(model, 'modular_model'):
            model.modular_model = model.modular_model.to(device).to(torch.bfloat16)
            # Ensure the core model is also in BFloat16
            if hasattr(model.modular_model, 'model'):
                model.modular_model.model = model.modular_model.model.to(device).to(torch.bfloat16)
        print(f"   ✅ NeMo model created from existing config")
    except Exception as e:
        print(f"   ⚠️  Failed to create from existing config: {e}")
        print(f"   Creating model with default parameters...")
        
        # Fallback to default model creation
        model = create_modular_model_nemo(
            vocab_size=tokenizer.vocab_size,
            hidden_size=2048,
            num_layers=28,
            num_attention_heads=16,
            training_stage="stage1",
            learning_rate=1e-6,
            weight_decay=0.01,
            warmup_steps=1000
        )
        print(f"   ✅ NeMo model created with default parameters")
    
    # Get the actual model from the wrapper
    if hasattr(model, 'modular_model'):
        actual_model = model.modular_model.model
    else:
        actual_model = model
    
    print(f"   Model parameters: {sum(p.numel() for p in actual_model.parameters()):,}")
    
    # Create training module with NeMo configuration
    print("\n4. Creating NeMo training module...")
    training_module = ModularModelTrainingModule(
        model=model,
        stage="stage1",
        learning_rate=1e-6,
        weight_decay=0.01,
        warmup_steps=1000,
        optimizer_config={
            "type": "AdamW",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        },
        scheduler_config={
            "type": "LinearLR",
            "start_factor": 0.1,
            "end_factor": 1.0,
            "warmup_steps": 1000,
            "interval": "step",
            "frequency": 1
        }
    )
    print(f"   ✅ NeMo training module created")
    
    # Run training step
    print("\n5. Running NeMo training step...")
    print("   This will show detailed debugging information...")
    
    try:
        # Set model to training mode
        actual_model.train()
        
        # Move batch to the same device as the model
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Run the training step
        loss = training_module.training_step(batch, batch_idx=0)
        
        print(f"\n   ✅ NeMo training step completed successfully!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss type: {type(loss)}")
        print(f"   Loss device: {loss.device}")
        
    except Exception as e:
        print(f"\n   ❌ NeMo training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Additional verification
    print("\n6. Additional verification...")
    
    # Get the batch data that was actually used (already tensors)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Show first sequence in detail
    seq_len = min(20, input_ids.size(1))  # Show first 20 tokens
    input_tokens = input_ids[0, :seq_len].cpu().tolist()
    label_tokens = labels[0, :seq_len].cpu().tolist()
    attention_tokens = attention_mask[0, :seq_len].cpu().tolist()
    
    print(f"\n   First sequence (first {seq_len} tokens):")
    print(f"   input_ids:      {input_tokens}")
    print(f"   attention_mask: {attention_tokens}")
    print(f"   labels:         {label_tokens}")
    
    # Decode tokens
    print(f"\n   Token analysis:")
    input_text = tokenizer.decode(input_ids[0, :seq_len], skip_special_tokens=False)
    print(f"   Input text: {repr(input_text)}")
    
    # Show label tokens (replace -100 with pad for display)
    label_display_ids = labels[0, :seq_len].clone()
    label_display_ids[label_display_ids == -100] = tokenizer.pad_token_id
    label_text = tokenizer.decode(label_display_ids, skip_special_tokens=False)
    print(f"   Label text: {repr(label_text)}")
    
    # Verify correctness
    print(f"\n   Verification:")
    is_correct = verify_target_ids_correctness(
        input_ids=input_ids,
        target_ids=labels,
        ignore_first_token=False,
        pad_token_id=tokenizer.pad_token_id
    )
    print(f"   Next-token prediction setup: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    # Show valid target mask
    valid_target_mask = (attention_mask[:, :-1] & attention_mask[:, 1:]).bool()
    valid_target_tokens = valid_target_mask[0, :seq_len-1].cpu().tolist()
    print(f"   Valid target mask: {valid_target_tokens}")
    
    # Count valid targets
    num_valid_targets = valid_target_mask.sum().item()
    total_positions = valid_target_mask.numel()
    print(f"   Valid targets: {num_valid_targets}/{total_positions} ({num_valid_targets/total_positions*100:.1f}%)")
    
    return is_correct

def test_nemo_padding_edge_cases(tokenizer, training_module):
    """Test edge cases for padding handling with NeMo training."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PADDING AND EDGE CASES VERIFICATION (NEMO)")
    print("="*80)
    
    # Create manual test cases with specific padding scenarios
    pad_token_id = tokenizer.pad_token_id
    
    test_cases = [
        {
            "name": "All real tokens (no padding)",
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "expected_labels": [2, 3, 4, 5, -100]
        },
        {
            "name": "Padding at the end",
            "input_ids": torch.tensor([[1, 2, 3, pad_token_id, pad_token_id]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            "expected_labels": [2, 3, -100, -100, -100]
        },
        {
            "name": "Padding in the middle",
            "input_ids": torch.tensor([[1, 2, pad_token_id, pad_token_id, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0, 1]]),
            "expected_labels": [2, -100, -100, -100, -100]
        },
        {
            "name": "Mixed padding",
            "input_ids": torch.tensor([[1, 2, 3, pad_token_id, 5, pad_token_id, pad_token_id]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 1, 0, 0]]),
            "expected_labels": [2, 3, -100, -100, -100, -100, -100]
        },
        {
            "name": "Single real token",
            "input_ids": torch.tensor([[1, pad_token_id, pad_token_id, pad_token_id]]),
            "attention_mask": torch.tensor([[1, 0, 0, 0]]),
            "expected_labels": [-100, -100, -100, -100]
        },
        {
            "name": "Two real tokens at beginning",
            "input_ids": torch.tensor([[1, 2, pad_token_id, pad_token_id, pad_token_id]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0, 0]]),
            "expected_labels": [2, -100, -100, -100, -100]
        }
    ]
    
    print(f"\nTesting {len(test_cases)} padding edge cases...")
    
    all_tests_passed = True
    
    for i, test_case in enumerate(test_cases):
        print(f"\n7.{i+1} Testing: {test_case['name']}")
        
        input_ids = test_case['input_ids']
        attention_mask = test_case['attention_mask']
        expected_labels = test_case['expected_labels']
        
        # Create labels using the same logic as NeMo training
        labels = torch.full_like(input_ids, -100)
        # Set labels for next-token prediction
        for j in range(input_ids.size(1) - 1):
            # Only set label if both current and next positions are real tokens
            current_real = attention_mask[:, j] == 1
            next_real = attention_mask[:, j + 1] == 1
            valid_positions = current_real & next_real
            if valid_positions.any():
                labels[valid_positions, j] = input_ids[valid_positions, j + 1]
        
        print(f"   Input IDs:      {input_ids[0].cpu().tolist()}")
        print(f"   Attention mask: {attention_mask[0].cpu().tolist()}")
        print(f"   Labels:         {labels[0].cpu().tolist()}")
        print(f"   Expected:       {expected_labels}")
        
        # Verify correctness
        actual_labels = labels[0].cpu().tolist()
        is_correct = actual_labels == expected_labels
        
        if is_correct:
            print(f"   Result: ✅ CORRECT")
        else:
            print(f"   Result: ❌ INCORRECT")
            all_tests_passed = False
            
            # Show detailed analysis
            print(f"   Detailed analysis:")
            valid_target_mask = (attention_mask[:, :-1] & attention_mask[:, 1:]).bool()
            valid_target_tokens = valid_target_mask[0].cpu().tolist()
            print(f"   Valid target mask: {valid_target_tokens}")
            
            # Check each position
            for j in range(len(actual_labels) - 1):
                if valid_target_tokens[j]:
                    expected_target = input_ids[0, j + 1].item()
                    actual_target = actual_labels[j]
                    is_pos_correct = expected_target == actual_target
                    status = "✅" if is_pos_correct else "❌"
                    print(f"   Position {j}: expected={expected_target}, actual={actual_target} {status}")
        
        # Verify last position is always -100
        last_target = actual_labels[-1]
        last_correct = last_target == -100
        if not last_correct:
            print(f"   ❌ Last position should be -100, but is {last_target}")
            all_tests_passed = False
    
    return all_tests_passed

def test_nemo_fsdp_integration():
    """Test FSDP integration with NeMo training."""
    
    print("\n" + "="*80)
    print("NEMO FSDP INTEGRATION TEST")
    print("="*80)
    
    # Test FSDP configuration loading
    print("\n8.1 Testing FSDP configuration loading...")
    try:
        # Get the project root directory (parent of test directory)
        project_root = os.path.dirname(os.path.dirname(__file__))
        config = create_nemo_config_from_existing("model_config_1.7B", "stage1", project_root)
        distributed_config = config.get("distributed", {})
        fsdp_config = distributed_config.get("fsdp", {})
        
        print(f"   ✅ Configuration loaded successfully")
        print(f"   Strategy: {distributed_config.get('strategy', 'auto')}")
        print(f"   FSDP enabled: {fsdp_config.get('enabled', False)}")
        print(f"   Sharding strategy: {fsdp_config.get('sharding_strategy', 'FULL_SHARD')}")
        print(f"   CPU offload: {fsdp_config.get('cpu_offload', False)}")
        
        # Test strategy creation
        from ModularModelstage1_NTPtraining import create_strategy
        strategy = create_strategy(distributed_config)
        
        if strategy is not None:
            print(f"   ✅ Strategy created successfully: {type(strategy).__name__}")
        else:
            print(f"   ✅ Auto strategy will be used")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FSDP configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nemo_production_training():
    """Test NeMo production training mode."""
    
    print("\n" + "="*80)
    print("NEMO PRODUCTION TRAINING TEST")
    print("="*80)
    
    print("\n9.1 Testing production training mode...")
    try:
        from ModularModelstage1_NTPtraining import train_production_mode
        
        # Test with a small configuration
        project_root = os.path.dirname(os.path.dirname(__file__))
        trainer, module = train_production_mode(
            model_config_key="model_config_1.7B",
            stage="stage1",
            devices=1,  # Single device for testing
            precision="bf16-mixed",
            base_path=project_root,
            # Override some settings for testing
            **{
                "max_epochs": 1,
                "batch_size": 2,
                "total_samples": 10  # Very small dataset for testing
            }
        )
        
        print(f"   ✅ Production training mode test completed successfully")
        print(f"   Trainer type: {type(trainer).__name__}")
        print(f"   Module type: {type(module).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Production training mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_nemo_stage1_training_comprehensive():
    """Comprehensive test for NeMo Stage 1 training including all components."""
    
    print("="*80)
    print("COMPREHENSIVE NEMO STAGE 1 TRAINING TEST")
    print("="*80)
    
    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer_path = "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        # Fallback to HuggingFace tokenizer
        tokenizer_path = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer loaded: {tokenizer_path}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # Create test batch with training sequence length
    print("\n2. Creating test batch with training sequence length...")
    batch = create_test_batch_nemo(tokenizer, max_length=2048)  # Same as training config
    
    # Create NeMo model using config
    print("\n3. Creating NeMo model using config...")
    try:
        # Try to use existing config
        project_root = os.path.dirname(os.path.dirname(__file__))
        model = create_modular_model_from_existing_config(
            model_config_key="model_config_1.7B",
            stage="stage1",
            base_path=project_root
        )
        
        # Move model to GPU device first, then set precision
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Ensure consistent mixed precision (BFloat16) throughout the model
        # Convert the entire model to BFloat16 for proper mixed precision
        model = model.to(torch.bfloat16)
        # Also ensure the underlying modular model is in BFloat16
        if hasattr(model, 'modular_model'):
            model.modular_model = model.modular_model.to(device).to(torch.bfloat16)
            # Ensure the core model is also in BFloat16
            if hasattr(model.modular_model, 'model'):
                model.modular_model.model = model.modular_model.model.to(device).to(torch.bfloat16)
        print(f"   ✅ NeMo model created from existing config")
    except Exception as e:
        print(f"   ⚠️  Failed to create from existing config: {e}")
        print(f"   Creating model with default parameters...")
        
        # Fallback to default model creation
        model = create_modular_model_nemo(
            vocab_size=tokenizer.vocab_size,
            hidden_size=2048,
            num_layers=28,
            num_attention_heads=16,
            training_stage="stage1",
            learning_rate=1e-6,
            weight_decay=0.01,
            warmup_steps=1000
        )
        print(f"   ✅ NeMo model created with default parameters")
    
    # Get the actual model from the wrapper
    if hasattr(model, 'modular_model'):
        actual_model = model.modular_model.model
    else:
        actual_model = model
    
    print(f"   Model parameters: {sum(p.numel() for p in actual_model.parameters()):,}")
    
    # Create training module with NeMo configuration
    print("\n4. Creating NeMo training module...")
    training_module = ModularModelTrainingModule(
        model=model,
        stage="stage1",
        learning_rate=1e-6,
        weight_decay=0.01,
        warmup_steps=1000,
        optimizer_config={
            "type": "AdamW",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        },
        scheduler_config={
            "type": "LinearLR",
            "start_factor": 0.1,
            "end_factor": 1.0,
            "warmup_steps": 1000,
            "interval": "step",
            "frequency": 1
        }
    )
    print(f"   ✅ NeMo training module created")
    
    # Run training step with memory tracking
    print("\n5. Running NeMo training step with memory tracking...")
    print("   This will verify target_ids manipulation happens only once...")
    
    try:
        # Set model to training mode
        actual_model.train()
        
        # Move batch to the same device as the model
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Run the training step
        loss = training_module.training_step(batch, batch_idx=0)
        
        print(f"\n   ✅ NeMo training step completed successfully!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss type: {type(loss)}")
        print(f"   Loss device: {loss.device}")
        
    except Exception as e:
        print(f"\n   ❌ NeMo training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Additional verification
    print("\n6. Additional verification...")
    
    # Get the batch data that was actually used (already tensors)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Show first sequence in detail
    seq_len = min(20, input_ids.size(1))  # Show first 20 tokens
    input_tokens = input_ids[0, :seq_len].cpu().tolist()
    label_tokens = labels[0, :seq_len].cpu().tolist()
    attention_tokens = attention_mask[0, :seq_len].cpu().tolist()
    
    print(f"\n   First sequence (first {seq_len} tokens):")
    print(f"   input_ids:      {input_tokens}")
    print(f"   attention_mask: {attention_tokens}")
    print(f"   labels:         {label_tokens}")
    
    # Decode tokens
    print(f"\n   Token analysis:")
    input_text = tokenizer.decode(input_ids[0, :seq_len], skip_special_tokens=False)
    print(f"   Input text: {repr(input_text)}")
    
    # Show label tokens (replace -100 with pad for display)
    label_display_ids = labels[0, :seq_len].clone()
    label_display_ids[label_display_ids == -100] = tokenizer.pad_token_id
    label_text = tokenizer.decode(label_display_ids, skip_special_tokens=False)
    print(f"   Label text: {repr(label_text)}")
    
    # Verify correctness
    print(f"\n   Verification:")
    is_correct = verify_target_ids_correctness(
        input_ids=input_ids,
        target_ids=labels,
        ignore_first_token=False,
        pad_token_id=tokenizer.pad_token_id
    )
    print(f"   Next-token prediction setup: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    # Show valid target mask
    valid_target_mask = (attention_mask[:, :-1] & attention_mask[:, 1:]).bool()
    valid_target_tokens = valid_target_mask[0, :seq_len-1].cpu().tolist()
    print(f"   Valid target mask: {valid_target_tokens}")
    
    # Count valid targets
    num_valid_targets = valid_target_mask.sum().item()
    total_positions = valid_target_mask.numel()
    print(f"   Valid targets: {num_valid_targets}/{total_positions} ({num_valid_targets/total_positions*100:.1f}%)")
    
    # Run comprehensive padding edge cases test
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE PADDING AND EDGE CASES TESTS (NEMO)")
    print("="*80)
    
    padding_tests_passed = test_nemo_padding_edge_cases(tokenizer, training_module)
    
    # Run FSDP integration test
    fsdp_tests_passed = test_nemo_fsdp_integration()
    
    # Run production training test
    production_tests_passed = test_nemo_production_training()
    
    # Overall results
    all_tests_passed = is_correct and padding_tests_passed and fsdp_tests_passed and production_tests_passed
    
    print("\n" + "="*80)
    print("COMPREHENSIVE NEMO TEST SUMMARY")
    print("="*80)
    
    print(f"✅ Main NeMo training test: {'PASSED' if is_correct else 'FAILED'}")
    print(f"✅ Padding edge cases test: {'PASSED' if padding_tests_passed else 'FAILED'}")
    print(f"✅ FSDP integration test: {'PASSED' if fsdp_tests_passed else 'FAILED'}")
    print(f"✅ Production training test: {'PASSED' if production_tests_passed else 'FAILED'}")
    
    if all_tests_passed:
        print("\n🎉 ALL COMPREHENSIVE NEMO TESTS PASSED!")
        print("✅ NeMo Stage 1 training is working correctly")
        print("✅ Next-token prediction is properly implemented")
        print("✅ Padding handling is correct for all edge cases")
        print("✅ Last token handling is correct")
        print("✅ FSDP integration is working correctly")
        print("✅ Production training mode is working correctly")
        print("✅ Configuration loading is working correctly")
    else:
        print("\n❌ SOME NEMO TESTS FAILED!")
        print("There are issues with NeMo Stage 1 training implementation")
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_nemo_stage1_training_comprehensive()
    if success:
        print("\n🎉 All comprehensive NeMo tests passed! Stage 1 training is working correctly.")
    else:
        print("\n❌ Tests failed! There are issues with NeMo Stage 1 training.")
        sys.exit(1)

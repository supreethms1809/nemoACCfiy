# NeMo Training Tests

This document explains how to use the comprehensive test suite for NeMo training.

## Overview

The NeMo training tests verify that the training pipeline works correctly with:
- Next-token prediction setup
- Padding handling
- FSDP integration
- Production training mode
- Configuration loading

## Test Files

### `test_comprehensive.py`
The main comprehensive test file that includes:
- **Basic NeMo Training Test**: Verifies the core training functionality
- **Padding Edge Cases Test**: Tests various padding scenarios
- **FSDP Integration Test**: Verifies FSDP configuration and strategy creation
- **Production Training Test**: Tests the full production training pipeline

### `run_nemo_tests.py`
A test runner script that provides easy access to different test types.

## Running Tests

### 1. Run All Tests
```bash
python run_nemo_tests.py --test all
```

### 2. Run Specific Tests
```bash
# Comprehensive test (includes all sub-tests)
python run_nemo_tests.py --test comprehensive

# Basic training test only
python run_nemo_tests.py --test basic

# FSDP integration test only
python run_nemo_tests.py --test fsdp

# Production training test only
python run_nemo_tests.py --test production
```

### 3. Run with Verbose Output
```bash
python run_nemo_tests.py --test comprehensive --verbose
```

### 4. Run Tests Directly
```bash
# Run the comprehensive test directly
python test/test_comprehensive.py
```

## Test Components

### 1. Basic NeMo Training Test
- Creates a NeMo model using the configuration system
- Tests the `ModularModelTrainingModule` with proper optimizer and scheduler setup
- Verifies next-token prediction is working correctly
- Tests with real tokenizer and model configurations

### 2. Padding Edge Cases Test
Tests various padding scenarios:
- All real tokens (no padding)
- Padding at the end
- Padding in the middle
- Mixed padding patterns
- Single real token
- Two real tokens at beginning

### 3. FSDP Integration Test
- Loads FSDP configuration from `config.yaml`
- Tests strategy creation with different FSDP options
- Verifies configuration loading works correctly
- Tests both enabled and disabled FSDP scenarios

### 4. Production Training Test
- Tests the full `train_production_mode` function
- Uses real configuration loading
- Tests with small datasets for quick verification
- Verifies the complete training pipeline

## Configuration Requirements

The tests expect the following configuration structure in `configs/config.yaml`:

```yaml
training_stages:
  stage1:
    model:
      model_config_key: "model_config_1.8B"
      tokenizer_path: "tokenizers/qwen3-coder-30b-a3b-instruct-custom"
    
    training:
      epochs: 2
      batch_size: 8
      sequence_length: 2048
      learning_rate: 1e-6
      mixed_precision: "bf16"
    
    distributed:
      strategy: "auto"
      num_nodes: 1
      devices: "auto"
      fsdp:
        enabled: false
        sharding_strategy: "FULL_SHARD"
        cpu_offload: false
        activation_checkpointing: true
    
    optimizer:
      type: "AdamW"
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8
    
    scheduler:
      type: "LinearLR"
      start_factor: 0.1
      end_factor: 1.0
      warmup_steps: 1000
      interval: "step"
      frequency: 1
```

## Expected Output

### Successful Test Run
```
================================================================================
COMPREHENSIVE NEMO STAGE 1 TRAINING TEST
================================================================================

1. Loading tokenizer...
   ✅ Tokenizer loaded: tokenizers/qwen3-coder-30b-a3b-instruct-custom
   Vocab size: 32000
   Pad token: '<|endoftext|>' (ID: 151643)
   EOS token: '<|endoftext|>' (ID: 151643)

2. Creating test batch with training sequence length...
   ✅ Test batch created

3. Creating NeMo model using config...
   ✅ NeMo model created from existing config
   Model parameters: 1,234,567,890

4. Creating NeMo training module...
   ✅ NeMo training module created

5. Running NeMo training step with memory tracking...
   ✅ NeMo training step completed successfully!
   Loss: 8.2341
   Loss type: <class 'torch.Tensor'>
   Loss device: cuda:0

6. Additional verification...
   Next-token prediction setup: ✅ CORRECT
   Valid targets: 2047/2047 (100.0%)

================================================================================
COMPREHENSIVE PADDING AND EDGE CASES TESTS (NEMO)
================================================================================

7.1 Testing: All real tokens (no padding)
   Result: ✅ CORRECT

7.2 Testing: Padding at the end
   Result: ✅ CORRECT

...

================================================================================
COMPREHENSIVE NEMO TEST SUMMARY
================================================================================

✅ Main NeMo training test: PASSED
✅ Padding edge cases test: PASSED
✅ FSDP integration test: PASSED
✅ Production training test: PASSED

🎉 ALL COMPREHENSIVE NEMO TESTS PASSED!
✅ NeMo Stage 1 training is working correctly
✅ Next-token prediction is properly implemented
✅ Padding handling is correct for all edge cases
✅ Last token handling is correct
✅ FSDP integration is working correctly
✅ Production training mode is working correctly
✅ Configuration loading is working correctly
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Tokenizer Not Found**: The test will fallback to HuggingFace tokenizer if local tokenizer is not available
3. **Config Not Found**: Ensure `configs/config.yaml` exists and has the correct structure
4. **CUDA Out of Memory**: The tests use small models and batches, but you can reduce batch size if needed

### Debug Mode

Enable verbose logging to see detailed information:
```bash
python run_nemo_tests.py --test comprehensive --verbose
```

### Manual Testing

You can also run individual test functions directly:
```python
from test.test_comprehensive import test_nemo_stage1_training
result = test_nemo_stage1_training()
print(f"Test result: {result}")
```

## Integration with CI/CD

The tests can be integrated into CI/CD pipelines:

```bash
# Run tests and exit with error code if any fail
python run_nemo_tests.py --test all
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi
```

## Performance Notes

- Tests are designed to run quickly with small models and datasets
- Memory usage is minimized for testing environments
- FSDP tests verify configuration without requiring multiple GPUs
- Production tests use minimal data to verify the pipeline works

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Add comprehensive error handling
3. Include verbose output for debugging
4. Update this README with new test descriptions
5. Ensure tests can run in both single-GPU and multi-GPU environments

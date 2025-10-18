# NeMo Training Tests

This directory contains comprehensive tests for the NeMo training system.

## Files

- **`test_comprehensive.py`**: Main comprehensive test file that verifies all aspects of NeMo training
- **`run_nemo_tests.py`**: Test runner script for easy execution of different test types

## Quick Start

```bash
# Run all tests
python run_nemo_tests.py --test all

# Run comprehensive test
python run_nemo_tests.py --test comprehensive

# Run test directly
python test_comprehensive.py
```

## Test Coverage

The comprehensive test covers:
- ✅ NeMo model creation and configuration loading
- ✅ Training module with optimizer and scheduler setup
- ✅ Next-token prediction correctness
- ✅ Padding handling for all edge cases
- ✅ FSDP integration and configuration
- ✅ Production training pipeline
- ✅ Memory efficiency and in-place operations

## Requirements

- PyTorch Lightning
- Transformers
- NeMo framework components
- Proper configuration in `../configs/config.yaml`

For detailed documentation, see `../NEMO_TESTS_README.md`.

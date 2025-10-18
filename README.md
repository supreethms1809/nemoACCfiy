# nemoACCfiy

A comprehensive NeMo-based training framework for modular language models with FSDP support.

## Quick Start

### Running Tests
```bash
# Run all tests
python run_tests.py --test all

# Run comprehensive test
python run_tests.py --test comprehensive

# Run with verbose output
python run_tests.py --test fsdp --verbose
```

### Training
```bash
# Production training with FSDP
python fsdp_example.py --fsdp-config multi_node_fsdp --devices 8

# Basic training
python -m src.nemo.ModularModelstage1_NTPtraining --mode production --stage stage1
```

## Documentation

- [FSDP Configuration](FSDP_README.md) - Multi-node multi-GPU training setup
- [Test Documentation](NEMO_TESTS_README.md) - Comprehensive test suite
- [Test Directory](test/README.md) - Test files and usage
#!/usr/bin/env python3
"""
Test script for NeMo Megatron training implementation

This script tests the NeMo Megatron training components to ensure they work correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to system path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_megatron_imports():
    """Test if NeMo Megatron components can be imported."""
    logger.info("Testing NeMo Megatron imports...")
    
    try:
        from src.nemo.megatron_data_loader import MegatronDataLoader, create_megatron_data_loader
        logger.info("✅ MegatronDataLoader imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import MegatronDataLoader: {e}")
        return False
    
    try:
        from src.nemo.megatron_training import MegatronTrainingModule, train_megatron_mode
        logger.info("✅ MegatronTrainingModule imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import MegatronTrainingModule: {e}")
        return False
    
    try:
        from src.nemo.megatron_config_loader import MegatronConfigLoader, create_megatron_config_from_existing
        logger.info("✅ MegatronConfigLoader imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import MegatronConfigLoader: {e}")
        return False
    
    return True


def test_megatron_config_loading():
    """Test Megatron configuration loading."""
    logger.info("Testing Megatron configuration loading...")
    
    try:
        from src.nemo.megatron_config_loader import create_megatron_config_from_existing
        
        # Test configuration loading
        config = create_megatron_config_from_existing(
            model_config_key="model_config_1.7B",
            stage="stage1",
            config_path="configs/config.yaml"
        )
        
        # Check required fields
        required_fields = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "learning_rate", "max_steps", "batch_size", "data_path", "tokenizer_path"
        ]
        
        for field in required_fields:
            if field not in config:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        logger.info("✅ Megatron configuration loaded successfully")
        logger.info(f"   Model: {config['hidden_size']} hidden, {config['num_hidden_layers']} layers")
        logger.info(f"   Training: {config['max_steps']} steps, batch size {config['batch_size']}")
        logger.info(f"   Data: {config['data_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load Megatron configuration: {e}")
        return False


def test_megatron_data_loader():
    """Test Megatron data loader initialization."""
    logger.info("Testing Megatron data loader initialization...")
    
    try:
        from src.nemo.megatron_data_loader import create_megatron_data_loader
        
        # Test data loader creation (without actual data files)
        data_loader = create_megatron_data_loader(
            data_path="./test_data",
            tokenizer_path="tokenizers/qwen3-coder-30b-a3b-instruct-custom",
            max_length=2048,
            stage="stage1"
        )
        
        logger.info("✅ Megatron data loader created successfully")
        logger.info(f"   Data path: {data_loader.data_path}")
        logger.info(f"   Max length: {data_loader.max_length}")
        logger.info(f"   Stage: {data_loader.stage}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create Megatron data loader: {e}")
        return False


def test_megatron_training_module():
    """Test Megatron training module initialization."""
    logger.info("Testing Megatron training module initialization...")
    
    try:
        from src.nemo.megatron_training import MegatronTrainingModule
        from src.nemo.nemo_wrapper import create_modular_model_nemo
        
        # Create a simple model for testing
        model = create_modular_model_nemo(
            stage="stage1",
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4
        )
        
        # Create training module
        training_module = MegatronTrainingModule(
            model=model,
            stage="stage1",
            learning_rate=1e-4,
            max_steps=1000
        )
        
        logger.info("✅ Megatron training module created successfully")
        logger.info(f"   Stage: {training_module.stage}")
        logger.info(f"   Learning rate: {training_module.learning_rate}")
        logger.info(f"   Max steps: {training_module.max_steps}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create Megatron training module: {e}")
        return False


def test_megatron_integration():
    """Test integration with main training script."""
    logger.info("Testing Megatron integration with main training script...")
    
    try:
        from src.nemo.ModularModelstage1_NTPtraining import train_megatron_mode_wrapper
        
        logger.info("✅ Megatron training wrapper imported successfully")
        logger.info("   Available for use with --mode megatron")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to import Megatron training wrapper: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting NeMo Megatron training tests...")
    
    tests = [
        ("Import Test", test_megatron_imports),
        ("Config Loading Test", test_megatron_config_loading),
        ("Data Loader Test", test_megatron_data_loader),
        ("Training Module Test", test_megatron_training_module),
        ("Integration Test", test_megatron_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! NeMo Megatron training is ready to use.")
        logger.info("\nUsage:")
        logger.info("  python train.py --mode megatron --stage stage1 --model_config model_config_1.7B")
        logger.info("  python src/nemo/ModularModelstage1_NTPtraining.py --mode megatron --stage stage1")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

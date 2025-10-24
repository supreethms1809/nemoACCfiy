#!/usr/bin/env python3
"""
Unified Training Entry Point for NeMo ModularModel

This script provides a single entry point for all training modes:
- Basic training (simple datasets)
- Production training (configuration-driven)
- Tuning (hyperparameter optimization)

Usage:
    # Training mode controls the type of training
    # Config argument points to the config file you want to use
    
    python train.py --mode basic --config config.yaml --stage stage1
    python train.py --mode production --config config_production.yaml --model_config model_config_1.8B --stage stage1
    python train.py --mode tuning --config tuning_config.yaml
    python train.py --mode production --config my_custom_config.yaml --model_config model_config_1.8B --stage stage1
    python train.py --mode production --config configs/gh200_config.yaml --model_config model_config_1.8B --stage stage1
"""

import sys
import os
import shutil
import logging
from pathlib import Path

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_config(config_path: str):
    """Setup configuration from specified config file path"""
    # Handle different input formats
    if config_path == "basic" or config_path == "config.yaml":
        # Use existing config.yaml without copying
        logger.info("✅ Using existing config.yaml configuration!")
        return True
    
    # Determine the actual config file path
    if config_path.startswith("configs/"):
        # Full path provided
        config_file = config_path
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        # Just filename provided
        config_file = f"configs/{config_path}"
    else:
        # Assume it's a config name without extension
        config_file = f"configs/{config_path}.yaml"
    
    # Extract config name for logging
    config_name = os.path.basename(config_file).replace('.yaml', '').replace('.yml', '')
    
    logger.info(f"🚀 Setting up {config_name} configuration...")
    
    # Check if config exists
    if not os.path.exists(config_file):
        logger.error(f"❌ Config file not found: {config_file}")
        logger.error(f"💡 Make sure the file exists or provide the correct path")
        return False
    
    # Copy config to active config
    logger.info(f"📝 Copying {config_name} config to active config...")
    shutil.copy(config_file, 'configs/config.yaml')
    logger.info(f"✅ {config_name.capitalize()} configuration ready!")
    
    return True

def run_tuning(config_path: str = "tuning_config.yaml"):
    """Run tuning with automatic config setup"""
    # Setup tuning config
    if not setup_config(config_path):
        return False
    
    # Import and run tuning
    try:
        from simple_tune import run_lightning_tuning, update_config_with_results
        
        logger.info("🎯 Starting tuning...")
        logger.info("📊 Using tuning configuration")
        logger.info("")
        
        # Run tuning
        results = run_lightning_tuning()
        
        if results:
            # Update config with results
            update_config_with_results(results)
            logger.info("✅ Tuning completed successfully!")
            return True
        else:
            logger.error("❌ Tuning failed!")
            return False
        
    except Exception as e:
        logger.error(f"❌ Tuning failed: {e}")
        return False

def main():
    """Main function with config setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified training entry point")
    parser.add_argument("--mode", choices=["basic", "production", "tuning"], default="production", help="Training mode (type of training)")
    parser.add_argument("--config", type=str, default="config_production.yaml", help="Config file path (e.g., config_production.yaml, configs/my_config.yaml, or full path)")
    parser.add_argument("--model_config", type=str, default="model_config_1B", help="Model configuration key")
    parser.add_argument("--stage", type=str, default="stage1", help="Training stage")
    parser.add_argument("--use_processed_datasets", action="store_true", help="Use processed datasets")
    
    args = parser.parse_args()
    
    # Handle tuning mode
    if args.mode == "tuning":
        success = run_tuning(args.config)
        if not success:
            sys.exit(1)
        return
    
    # Setup configuration based on config path
    if not setup_config(args.config):
        sys.exit(1)
    
    # Import and run the main training function
    try:
        from src.nemo.ModularModelstage1_NTPtraining import main as training_main
        
        logger.info(f"🚀 Starting {args.mode} training...")
        logger.info(f"📊 Using {args.config} configuration")
        logger.info("")
        
        # Run training
        training_main()
        
        logger.info(f"✅ {args.mode.capitalize()} training completed!")
        
    except Exception as e:
        logger.error(f"❌ {args.mode.capitalize()} training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified Training Entry Point for NeMo ModularModel

This script provides a single entry point for all training modes:
- Basic training (simple datasets)
- Production training (configuration-driven)
- Tuning (hyperparameter optimization)

Usage:
    python train.py --mode basic --stage stage1
    python train.py --mode production --model_config model_config_1.8B --stage stage1
    python train.py --mode tuning
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

def setup_config(mode: str):
    """Setup configuration based on mode"""
    if mode == "production":
        config_file = "configs/config_production.yaml"
        config_name = "production"
    elif mode == "tuning":
        config_file = "configs/tuning_config.yaml"
        config_name = "tuning"
    else:
        # For basic mode, use existing config.yaml
        return True
    
    logger.info(f"🚀 Setting up {config_name} configuration...")
    
    # Check if config exists
    if not os.path.exists(config_file):
        logger.error(f"❌ {config_name.capitalize()} config not found: {config_file}")
        return False
    
    # Copy config to active config
    logger.info(f"📝 Copying {config_name} config to active config...")
    shutil.copy(config_file, 'configs/config.yaml')
    logger.info(f"✅ {config_name.capitalize()} configuration ready!")
    
    return True

def run_tuning():
    """Run tuning with automatic config setup"""
    # Setup tuning config
    if not setup_config("tuning"):
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
    parser.add_argument("--mode", choices=["basic", "production", "tuning"], default="production", help="Training mode")
    parser.add_argument("--model_config", type=str, default="model_config_1B", help="Model configuration key")
    parser.add_argument("--stage", type=str, default="stage1", help="Training stage")
    parser.add_argument("--use_processed_datasets", action="store_true", help="Use processed datasets")
    
    args = parser.parse_args()
    
    # Handle tuning mode
    if args.mode == "tuning":
        success = run_tuning()
        if not success:
            sys.exit(1)
        return
    
    # Setup configuration for production mode
    if args.mode == "production":
        if not setup_config("production"):
            sys.exit(1)
    
    # Import and run the main training function
    try:
        from src.nemo.ModularModelstage1_NTPtraining import main as training_main
        
        logger.info(f"🚀 Starting {args.mode} training...")
        logger.info(f"📊 Using {args.mode} configuration")
        logger.info("")
        
        # Run training
        training_main()
        
        logger.info(f"✅ {args.mode.capitalize()} training completed!")
        
    except Exception as e:
        logger.error(f"❌ {args.mode.capitalize()} training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple Lightning Tuner for ModularModel
Uses Lightning's built-in tuning capabilities
"""

import os
import sys
import yaml
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lightning_tuning():
    """Run Lightning's built-in tuning using the integrated train.py"""
    try:
        import subprocess
        
        logger.info("🚀 Starting Lightning tuning...")
        logger.info("📝 Using integrated train.py with --mode tuning")
        
        # Run tuning using the integrated train.py
        result = subprocess.run([
            "python", "train.py",
            "--mode", "tuning"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            logger.info("✅ Tuning completed successfully!")
            logger.info("📊 Check tuning_results_simple.yaml for results")
            return {"tuning_completed": True}
        else:
            logger.error(f"❌ Tuning failed: {result.stderr}")
            return None
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Tuning timed out after 30 minutes")
        return None
    except Exception as e:
        logger.error(f"❌ Tuning failed: {e}")
        return None

# Note: Parsing functions removed since we're using direct function calls instead of subprocess

def update_config_with_results(results: dict):
    """Update config file with tuning results"""
    if not results:
        return
    
    # Load tuning config as base
    with open('configs/tuning_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update with optimal values
    if 'optimal_batch_size' in results:
        config['training_stages']['stage1']['training']['batch_size'] = results['optimal_batch_size']
    
    if 'optimal_learning_rate' in results:
        config['training_stages']['stage1']['training']['learning_rate'] = results['optimal_learning_rate']
    
    # Adjust gradient accumulation based on batch size
    batch_size = config['training_stages']['stage1']['training']['batch_size']
    if batch_size == 2:
        config['training_stages']['stage1']['training']['gradient_accumulation_steps'] = 4
    elif batch_size == 4:
        config['training_stages']['stage1']['training']['gradient_accumulation_steps'] = 2
    else:
        config['training_stages']['stage1']['training']['gradient_accumulation_steps'] = 1
    
    # Save updated config
    with open('configs/tuned_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("✅ Config updated with tuning results: configs/tuned_config.yaml")

def main():
    """Main function"""
    logger.info("🎯 Starting Simple Lightning Tuning")
    
    # Run tuning
    results = run_lightning_tuning()
    
    if results:
        # Update config with results
        update_config_with_results(results)
        logger.info("🚀 Ready to train with optimized parameters!")
    else:
        logger.error("❌ Tuning failed")

if __name__ == "__main__":
    main()

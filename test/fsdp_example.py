#!/usr/bin/env python3
"""
Example script demonstrating how to use FSDP for multi-node multi-GPU training.

This script shows how to:
1. Configure FSDP in the config.yaml
2. Run training with FSDP strategy
3. Use different FSDP configurations for different scenarios
"""

import os
import sys
import yaml
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src" / "nemo"))

from src.nemo.ModularModelstage1_NTPtraining import train_production_mode


def update_config_for_fsdp(config_path: str, fsdp_config_name: str = "multi_node_fsdp"):
    """
    Update the config.yaml to use a specific FSDP configuration.
    
    Args:
        config_path: Path to the config.yaml file
        fsdp_config_name: Name of the FSDP configuration to use from distributed_examples
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the FSDP configuration from distributed_examples
    if fsdp_config_name in config.get("distributed_examples", {}):
        fsdp_config = config["distributed_examples"][fsdp_config_name]
        
        # Update both stage1 and stage2 with the FSDP configuration
        for stage in ["stage1", "stage2"]:
            if stage in config.get("training_stages", {}):
                config["training_stages"][stage]["distributed"] = fsdp_config
        
        # Write the updated config back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated config.yaml to use {fsdp_config_name} configuration")
        print(f"   Strategy: {fsdp_config.get('strategy', 'auto')}")
        print(f"   Nodes: {fsdp_config.get('num_nodes', 1)}")
        print(f"   Devices: {fsdp_config.get('devices', 'auto')}")
        if fsdp_config.get('fsdp', {}).get('enabled', False):
            print(f"   FSDP enabled with sharding: {fsdp_config['fsdp'].get('sharding_strategy', 'FULL_SHARD')}")
    else:
        print(f"❌ FSDP configuration '{fsdp_config_name}' not found in distributed_examples")


def run_fsdp_training(
    stage: str = "stage1",
    model_config: str = "model_config_1.8B",
    fsdp_config: str = "multi_node_fsdp",
    devices: int = None,
    num_nodes: int = None
):
    """
    Run training with FSDP configuration.
    
    Args:
        stage: Training stage ("stage1" or "stage2")
        model_config: Model configuration key
        fsdp_config: FSDP configuration name from distributed_examples
        devices: Override number of devices (if None, uses config value)
        num_nodes: Override number of nodes (if None, uses config value)
    """
    config_path = "configs/config.yaml"
    
    # Update config with FSDP settings
    update_config_for_fsdp(config_path, fsdp_config)
    
    # Override devices and num_nodes if provided
    if devices is not None or num_nodes is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if stage in config.get("training_stages", {}):
            if devices is not None:
                config["training_stages"][stage]["distributed"]["devices"] = devices
            if num_nodes is not None:
                config["training_stages"][stage]["distributed"]["num_nodes"] = num_nodes
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"🚀 Starting {stage} training with FSDP...")
    print(f"   Model config: {model_config}")
    print(f"   FSDP config: {fsdp_config}")
    
    # Run training
    try:
        trainer, module = train_production_mode(
            model_config_key=model_config,
            stage=stage,
            devices=devices,
            precision="bf16-mixed"
        )
        print("✅ Training completed successfully!")
        return trainer, module
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def main():
    """Main function with example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FSDP Training Example")
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2"],
                       help="Training stage")
    parser.add_argument("--model-config", type=str, default="model_config_1.8B",
                       help="Model configuration key")
    parser.add_argument("--fsdp-config", type=str, default="multi_node_fsdp",
                       choices=["single_node_fsdp", "multi_node_fsdp", "multi_node_ddp", "large_scale_fsdp"],
                       help="FSDP configuration to use")
    parser.add_argument("--devices", type=int, default=None,
                       help="Number of devices (overrides config)")
    parser.add_argument("--num-nodes", type=int, default=None,
                       help="Number of nodes (overrides config)")
    
    args = parser.parse_args()
    
    # Run training with FSDP
    run_fsdp_training(
        stage=args.stage,
        model_config=args.model_config,
        fsdp_config=args.fsdp_config,
        devices=args.devices,
        num_nodes=args.num_nodes
    )


if __name__ == "__main__":
    main()

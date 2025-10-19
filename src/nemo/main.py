#!/usr/bin/env python3
"""
Main entry point for NeMo ModularModel training and inference.

This script provides a unified interface for training and using the ModularModel
with NeMo integration.
"""

import argparse
import sys
import os
from typing import Optional

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nemo.nemo_wrapper import create_modular_model_nemo, ModularModelConfig, create_modular_model_from_existing_config

# Import unified training functionality
try:
    from src.nemo.ModularModelstage1_NTPtraining import train_basic_mode, train_production_mode, train_foundation_mode
    TRAINING_AVAILABLE = True
    FOUNDATION_TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    FOUNDATION_TRAINING_AVAILABLE = False
    print("Warning: Unified training not available. Training functionality disabled.")


def create_model(
    stage: str = "stage1",
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_attention_heads: int = 12,
    num_kv_heads: Optional[int] = None,
    intermediate_size: int = 3072,
    max_position_embeddings: int = 512,
    num_reasoning_vectors: int = 8,
    pool_type: str = "mean",
    tie_weights: bool = True,
    freeze_embedder_decoder: bool = True,
    embedder_checkpoint_path: Optional[str] = None,
    attention_type: str = "gqa",
    mlp_type: str = "gated",
    use_flash_attention: bool = True,
    use_existing_config: bool = False,
    model_config_key: str = "model_config_1.7B",
    base_path: Optional[str] = None,
    **kwargs
):
    """Create a ModularModel with NeMo integration."""
    if use_existing_config:
        print(f"Using existing configuration: {model_config_key}")
        return create_modular_model_from_existing_config(
            model_config_key=model_config_key,
            stage=stage,
            base_path=base_path
        )
    else:
        return create_modular_model_nemo(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_reasoning_vectors=num_reasoning_vectors,
            pool_type=pool_type,
            tie_weights=tie_weights,
            freeze_embedder_decoder=freeze_embedder_decoder,
            embedder_checkpoint_path=embedder_checkpoint_path,
            attention_type=attention_type,
            mlp_type=mlp_type,
            use_flash_attention=use_flash_attention,
            training_stage=stage,
            **kwargs
        )


def train_model(
    stage: str = "stage1",
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_attention_heads: int = 12,
    num_kv_heads: Optional[int] = None,
    intermediate_size: int = 3072,
    max_position_embeddings: int = 512,
    num_reasoning_vectors: int = 8,
    pool_type: str = "mean",
    tie_weights: bool = True,
    freeze_embedder_decoder: bool = True,
    embedder_checkpoint_path: Optional[str] = None,
    attention_type: str = "gqa",
    mlp_type: str = "gated",
    use_flash_attention: bool = True,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_epochs: int = 10,
    batch_size: int = 4,
    num_workers: int = 8,
    devices: int = 1,
    precision: str = "16-mixed",
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    log_every_n_steps: int = 10,
    val_check_interval: float = 1.0,
    save_top_k: int = 3,
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 5,
    output_dir: str = "./outputs",
    use_existing_config: bool = False,
    model_config_key: str = "model_config_1.7B",
    base_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    **kwargs
):
    """Train a ModularModel with NeMo integration."""
    if use_existing_config:
        print(f"Training with existing configuration: {model_config_key}")
        # Create model from existing config
        model = create_modular_model_from_existing_config(
            model_config_key=model_config_key,
            stage=stage,
            base_path=base_path
        )
        
        # Override training parameters from command line if provided
        training_kwargs = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'devices': devices,
            'precision': precision,
            'gradient_clip_val': gradient_clip_val,
            'accumulate_grad_batches': accumulate_grad_batches,
            'log_every_n_steps': log_every_n_steps,
            'val_check_interval': val_check_interval,
            'save_top_k': save_top_k,
            'monitor': monitor,
            'mode': mode,
            'patience': patience,
            'output_dir': output_dir,
            'resume_from_checkpoint': resume_from_checkpoint,
            **kwargs
        }
        
        return train_production_mode(
            model_config_key=model_config_key,
            stage=stage,
            **training_kwargs
        )
    else:
        return train_basic_mode(
            stage=stage,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_reasoning_vectors=num_reasoning_vectors,
            pool_type=pool_type,
            tie_weights=tie_weights,
            freeze_embedder_decoder=freeze_embedder_decoder,
            embedder_checkpoint_path=embedder_checkpoint_path,
            attention_type=attention_type,
            mlp_type=mlp_type,
            use_flash_attention=use_flash_attention,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_epochs=max_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            devices=devices,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        save_top_k=save_top_k,
        monitor=monitor,
        mode=mode,
        patience=patience,
        output_dir=output_dir,
        **kwargs
    )


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="NeMo ModularModel - Training and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a model
  python main.py create --stage stage1 --vocab_size 1000 --hidden_size 256

  # Train Stage 1 with custom datasets
  python main.py train --stage stage1 --vocab_size 1000 --hidden_size 256 --max_epochs 10

  # Train Stage 2 with custom datasets
  python main.py train --stage stage2 --vocab_size 1000 --hidden_size 256 --max_epochs 10

  # Train with NeMo foundation datasets
  python main.py foundation --stage stage1 --data_path ./data --max_epochs 3

  # Train with existing configuration
  python main.py foundation --use_existing_config --model_config_key model_config_micro --stage stage1

  # Test the integration
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a model')
    create_parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2'])
    create_parser.add_argument('--vocab_size', type=int, default=32000)
    create_parser.add_argument('--hidden_size', type=int, default=768)
    create_parser.add_argument('--num_layers', type=int, default=12)
    create_parser.add_argument('--num_attention_heads', type=int, default=12)
    create_parser.add_argument('--num_kv_heads', type=int, default=None)
    create_parser.add_argument('--intermediate_size', type=int, default=3072)
    create_parser.add_argument('--max_position_embeddings', type=int, default=512)
    create_parser.add_argument('--num_reasoning_vectors', type=int, default=8)
    create_parser.add_argument('--pool_type', type=str, default='mean')
    create_parser.add_argument('--tie_weights', action='store_true', default=True)
    create_parser.add_argument('--freeze_embedder_decoder', action='store_true', default=True)
    create_parser.add_argument('--embedder_checkpoint_path', type=str, default=None)
    create_parser.add_argument('--attention_type', type=str, default='gqa', choices=['gqa', 'vanilla'])
    create_parser.add_argument('--mlp_type', type=str, default='gated', choices=['mlp', 'gated'])
    create_parser.add_argument('--use_flash_attention', action='store_true', default=True)
    
    # Configuration options
    create_parser.add_argument('--use_existing_config', action='store_true', 
                              help='Use existing configuration from model_config and run_config')
    create_parser.add_argument('--model_config_key', type=str, default='model_config_1.7B',
                              help='Model configuration key from config.json')
    create_parser.add_argument('--base_path', type=str, default=None,
                              help='Base path to src directory (auto-detected if not provided)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    
    # Foundation train command
    foundation_parser = subparsers.add_parser('foundation', help='Train with NeMo foundation datasets')
    train_parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2'])
    train_parser.add_argument('--vocab_size', type=int, default=32000)
    train_parser.add_argument('--hidden_size', type=int, default=768)
    train_parser.add_argument('--num_layers', type=int, default=12)
    train_parser.add_argument('--num_attention_heads', type=int, default=12)
    train_parser.add_argument('--num_kv_heads', type=int, default=None)
    train_parser.add_argument('--intermediate_size', type=int, default=3072)
    train_parser.add_argument('--max_position_embeddings', type=int, default=512)
    train_parser.add_argument('--num_reasoning_vectors', type=int, default=8)
    train_parser.add_argument('--pool_type', type=str, default='mean')
    train_parser.add_argument('--tie_weights', action='store_true', default=True)
    train_parser.add_argument('--freeze_embedder_decoder', action='store_true', default=True)
    train_parser.add_argument('--embedder_checkpoint_path', type=str, default=None)
    train_parser.add_argument('--attention_type', type=str, default='gqa', choices=['gqa', 'vanilla'])
    train_parser.add_argument('--mlp_type', type=str, default='gated', choices=['mlp', 'gated'])
    train_parser.add_argument('--use_flash_attention', action='store_true', default=True)
    
    # Configuration options for training
    train_parser.add_argument('--use_existing_config', action='store_true', 
                              help='Use existing configuration from model_config and run_config')
    train_parser.add_argument('--model_config_key', type=str, default='model_config_1.7B',
                              help='Model configuration key from config.json')
    train_parser.add_argument('--base_path', type=str, default=None,
                              help='Base path to src directory (auto-detected if not provided)')
    train_parser.add_argument('--dataset_source', type=str, default='auto', 
                              choices=['auto', 'sample', 'huggingface'],
                              help='Dataset source: auto (use config), sample (local datasets), huggingface (HF datasets)')
    
    train_parser.add_argument('--learning_rate', type=float, default=1e-4)
    train_parser.add_argument('--weight_decay', type=float, default=0.01)
    train_parser.add_argument('--warmup_steps', type=int, default=1000)
    train_parser.add_argument('--max_epochs', type=int, default=10)
    train_parser.add_argument('--batch_size', type=int, default=4)
    train_parser.add_argument('--num_workers', type=int, default=8)
    train_parser.add_argument('--devices', type=int, default=1)
    train_parser.add_argument('--precision', type=str, default='16-mixed')
    train_parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    train_parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    train_parser.add_argument('--log_every_n_steps', type=int, default=10)
    train_parser.add_argument('--val_check_interval', type=float, default=1.0)
    train_parser.add_argument('--save_top_k', type=int, default=3)
    train_parser.add_argument('--monitor', type=str, default='val_loss')
    train_parser.add_argument('--mode', type=str, default='min', choices=['min', 'max'])
    train_parser.add_argument('--patience', type=int, default=5)
    train_parser.add_argument('--output_dir', type=str, default='./outputs')
    train_parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                              help='Path to checkpoint to resume training from')
    
    # Foundation training arguments
    foundation_parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2'])
    foundation_parser.add_argument('--use_existing_config', action='store_true', 
                                  help='Use existing configuration from model_config and run_config')
    foundation_parser.add_argument('--model_config_key', type=str, default='model_config_1.7B',
                                  help='Model configuration key from config.json')
    foundation_parser.add_argument('--base_path', type=str, default=None,
                                  help='Base path to src directory (auto-detected if not provided)')
    foundation_parser.add_argument('--dataset_source', type=str, default='auto', 
                                  choices=['auto', 'sample', 'huggingface'],
                                  help='Dataset source: auto (use config), sample (local datasets), huggingface (HF datasets)')
    
    # Dataset configuration
    foundation_parser.add_argument('--data_path', type=str, default='./data', help='Path to training data')
    foundation_parser.add_argument('--tokenizer_path', type=str, default='Qwen/Qwen3-Coder-480B-A35B-Instruct', help='Tokenizer path')
    foundation_parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    foundation_parser.add_argument('--use_mmap', action='store_true', default=True, help='Use memory mapping')
    foundation_parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    # Training configuration
    foundation_parser.add_argument('--learning_rate', type=float, default=1e-5)
    foundation_parser.add_argument('--weight_decay', type=float, default=0.01)
    foundation_parser.add_argument('--warmup_steps', type=int, default=100)
    foundation_parser.add_argument('--max_epochs', type=int, default=3)
    foundation_parser.add_argument('--batch_size', type=int, default=4)
    foundation_parser.add_argument('--num_workers', type=int, default=8)
    foundation_parser.add_argument('--devices', type=int, default=1)
    foundation_parser.add_argument('--precision', type=str, default='bf16-mixed')
    foundation_parser.add_argument('--mixed_precision', type=str, default='bf16')
    foundation_parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    foundation_parser.add_argument('--max_grad_norm', type=float, default=1.0)
    foundation_parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    foundation_parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    foundation_parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    foundation_parser.add_argument('--use_fsdp', action='store_true')
    
    # Logging and checkpointing
    foundation_parser.add_argument('--log_every_n_steps', type=int, default=10)
    foundation_parser.add_argument('--val_check_interval', type=float, default=1.0)
    foundation_parser.add_argument('--save_top_k', type=int, default=3)
    foundation_parser.add_argument('--monitor', type=str, default='val_loss')
    foundation_parser.add_argument('--mode', type=str, default='min', choices=['min', 'max'])
    foundation_parser.add_argument('--patience', type=int, default=5)
    foundation_parser.add_argument('--output_dir', type=str, default='./outputs')
    foundation_parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    foundation_parser.add_argument('--model_output_dir', type=str, default='./models')
    foundation_parser.add_argument('--model_output_name', type=str, default='nemo_foundation_model.ckpt')
    foundation_parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                                  help='Path to checkpoint to resume training from')
    
    # Wandb configuration
    foundation_parser.add_argument('--use_wandb', action='store_true', default=False)
    foundation_parser.add_argument('--wandb_project', type=str, default='accfiy')
    foundation_parser.add_argument('--wandb_entity', type=str, default=None)
    foundation_parser.add_argument('--wandb_run_name', type=str, default=None)
    foundation_parser.add_argument('--wandb_tags', type=str, nargs='+', default=['stage1', 'foundation'])
    foundation_parser.add_argument('--wandb_group', type=str, default=None)
    foundation_parser.add_argument('--wandb_notes', type=str, default=None)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the NeMo integration')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        print(f"Creating ModularModel for {args.stage}...")
        model = create_model(**vars(args))
        print(f"✅ Model created successfully!")
        print(f"Training stage: {model.modular_model.current_stage}")
        print(f"Model type: {model.modular_model.model_type}")
        trainable_params = model.get_trainable_parameters()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {len(trainable_params):,}")
        print(f"Total parameters: {total_params:,}")
        
    elif args.command == 'train':
        if not TRAINING_AVAILABLE:
            print("❌ Training functionality not available. Please install PyTorch Lightning.")
            sys.exit(1)
        print(f"Training ModularModel for {args.stage}...")
        model, trainer = train_model(**vars(args))
        print(f"✅ Training completed successfully!")
        
    elif args.command == 'foundation':
        if not FOUNDATION_TRAINING_AVAILABLE:
            print("❌ NeMo foundation training not available. Please check NeMo installation.")
            sys.exit(1)
        print(f"Training ModularModel with NeMo foundation datasets for {args.stage}...")
        model, trainer = train_foundation_mode(**vars(args))
        print(f"✅ NeMo foundation training completed successfully!")
        
    elif args.command == 'test':
        print("Running NeMo integration tests...")
        from test_nemo_integration import main as test_main
        success = test_main()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

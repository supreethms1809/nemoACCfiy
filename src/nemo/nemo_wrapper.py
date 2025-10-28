"""
NeMo Wrapper for ModularModel

This module provides a complete NeMo integration for the ModularModel,
including proper NeMo module registration, configuration management, and training integration.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import logging

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the optimized wrapper from optimized_models.py (shared by both NeMo and fallback)
try:
    from ..model.optimized_models import DecoderOnlyModel
    OPTIMIZED_MODEL_AVAILABLE = True
except ImportError:
    OPTIMIZED_MODEL_AVAILABLE = False
    DecoderOnlyModel = None

# NeMo imports
try:
    from nemo.core.classes import NeuralModule, typecheck, Loss, Exportable
    from nemo.core.neural_types import NeuralType, ChannelType, MaskType, Index, LogitsType, LossType
    from nemo.core.config import Config
    from nemo.utils import logging as nemo_logging
    NEMO_AVAILABLE = True
    
    # Define ModularModelNeMo when NeMo is available
    class ModularModelNeMo(NeuralModule):
        """NeMo-based implementation of ModularModelNeMo using the actual ModularModel."""
        
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            
            if OPTIMIZED_MODEL_AVAILABLE and DecoderOnlyModel is not None:
                # Stage 1 only: Use DecoderOnlyModel (decoder-only training)
                print(f"✅ [NeMo] Stage 1: Using DecoderOnlyModel wrapper from optimized_models.py")
                self.model = DecoderOnlyModel(
                    config=cfg,
                    vocab_size=cfg.vocab_size,
                    tie_weights=cfg.tie_weights
                )
            else:
                # Fallback to simple transformer if DecoderOnlyModel not available
                self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=cfg.hidden_size,
                        nhead=cfg.num_attention_heads,
                        dim_feedforward=cfg.intermediate_size,
                        dropout=cfg.hidden_dropout_prob,
                        batch_first=True
                    ),
                    num_layers=cfg.num_layers
                )
                self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
                self.model = None
            
        def forward(self, input_ids, attention_mask=None, labels=None, embed_input_ids=None, embed_attention_mask=None):
            if self.model is not None:
                # Use the actual ModularModel
                # For stage 1 training, embed_input_ids should be None
                logits = self.model(
                    input_ids=input_ids,
                    embed_input_ids=embed_input_ids,
                    attention_mask=attention_mask,
                    embed_attention_mask=embed_attention_mask
                )
            else:
                # Fallback implementation
                x = self.embedding(input_ids)
                x = self.transformer(x, src_key_padding_mask=attention_mask)
                logits = self.lm_head(x)
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return {"loss": loss, "logits": logits}
            
            return {"logits": logits}
        
        @property
        def input_types(self):
            return {}
        
        @property
        def output_types(self):
            return {}
        
        def named_modules(self, memo=None, prefix='', remove_duplicate=True):
            """Return all named modules for PyTorch Lightning compatibility."""
            if self.model is not None:
                # Use the actual ModularModel modules
                return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
            else:
                # Fallback to standard nn.Module implementation
                return super().named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
            
except ImportError:
    print("Warning: NeMo not available, using PyTorch fallback implementation")
    NEMO_AVAILABLE = False
    
    # Fallback classes
    class NeuralModule(nn.Module):
        pass
    
    class Exportable:
        pass
    
    def typecheck(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    
    class NeuralType:
        def __init__(self, *args, **kwargs):
            pass
    
    class ChannelType:
        pass
    
    class MaskType:
        pass
    
    class LogitsType:
        pass
    
    class LossType:
        pass
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class nemo_logging:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)
        
        @staticmethod
        def info(message):
            logging.info(message)
        
        @staticmethod
        def warning(message):
            logging.warning(message)
    
    # Fallback ModularModelNeMo class
    class ModularModelNeMo(nn.Module):
        """Fallback implementation of ModularModelNeMo when NeMo is not available."""
        
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            
            if OPTIMIZED_MODEL_AVAILABLE and DecoderOnlyModel is not None:
                # Stage 1 only: Use DecoderOnlyModel (decoder-only training)
                print(f"✅ [Fallback] Stage 1: Using DecoderOnlyModel wrapper from optimized_models.py")
                self.model = DecoderOnlyModel(
                    config=cfg,
                    vocab_size=cfg.vocab_size,
                    tie_weights=cfg.tie_weights
                )
            else:
                # Fallback to simple transformer if DecoderOnlyModel not available
                self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=cfg.hidden_size,
                        nhead=cfg.num_attention_heads,
                        dim_feedforward=cfg.intermediate_size,
                        dropout=cfg.hidden_dropout_prob,
                        batch_first=True
                    ),
                    num_layers=cfg.num_layers
                )
                self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
                self.model = None
            
        def forward(self, input_ids, attention_mask=None, labels=None, embed_input_ids=None, embed_attention_mask=None):
            if self.model is not None:
                # Use the actual ModularModel
                # For stage 1 training, embed_input_ids should be None
                logits = self.model(
                    input_ids=input_ids,
                    embed_input_ids=embed_input_ids,
                    attention_mask=attention_mask,
                    embed_attention_mask=embed_attention_mask
                )
            else:
                # Fallback implementation
                x = self.embedding(input_ids)
                x = self.transformer(x, src_key_padding_mask=attention_mask)
                logits = self.lm_head(x)
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return {"loss": loss, "logits": logits}
            
            return {"logits": logits}
        
        @property
        def input_types(self):
            return {}
        
        @property
        def output_types(self):
            return {}
        
        def named_modules(self, memo=None, prefix='', remove_duplicate=True):
            """Return all named modules for PyTorch Lightning compatibility."""
            if self.model is not None:
                # Use the actual ModularModel modules
                return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
            else:
                # Fallback to standard nn.Module implementation
                return super().named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
        
        @staticmethod
        def debug(message):
            logging.debug(message)

# Import your custom models
import sys
# Model imports are now handled by the project root path setup above

try:
    from src.model.embed_decoder_model import ModularModel
    from src.model.optimized_models import DecoderOnlyModel, EmbedderOnlyModel, create_optimized_model
except ImportError as e:
    print(f"Warning: Could not import model components: {e}")
    print("Make sure the model files are in the correct location")
    raise

# Import configuration loader
try:
    from src.nemo.config_loader import ConfigLoader, create_nemo_config_from_existing
except ImportError:
    # Silent fallback - config loader is optional
    ConfigLoader = None
    create_nemo_config_from_existing = None


class ModularModelConfig(Config):
    """NeMo configuration for the ModularModel."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Model architecture parameters
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_layers = kwargs.get('num_layers', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.num_kv_heads = kwargs.get('num_kv_heads', None)  # For GQA
        self.intermediate_size = kwargs.get('intermediate_size', 3072)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 512)
        self.dropout = kwargs.get('dropout', 0.1)
        self.attention_dropout = kwargs.get('attention_dropout', 0.1)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.layer_norm_epsilon = kwargs.get('layer_norm_epsilon', 1e-6)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)
        self.activation = kwargs.get('activation', 'gelu')
        self.use_cache = kwargs.get('use_cache', True)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.eos_token_id = kwargs.get('eos_token_id', 2)
        
        # ModularModel specific parameters
        self.pool_type = kwargs.get('pool_type', 'mean')
        self.num_reasoning_vectors = kwargs.get('num_reasoning_vectors', 8)
        self.tie_weights = kwargs.get('tie_weights', True)
        self.freeze_embedder_decoder = kwargs.get('freeze_embedder_decoder', True)
        self.embedder_checkpoint_path = kwargs.get('embedder_checkpoint_path', None)
        
        # Attention and MLP configuration
        self.attention_type = kwargs.get('attention_type', 'gqa')  # 'gqa' or 'vanilla'
        self.mlp_type = kwargs.get('mlp_type', 'mlp')  # 'mlp' or 'gated'
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.rotary_base = kwargs.get('rotary_base', 10000.0)
        
        # Training parameters
        self.training_stage = kwargs.get('training_stage', 'stage1')  # 'stage1' or 'stage2'
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        
        # Data parameters
        self.train_ds = kwargs.get('train_ds', {})
        self.validation_ds = kwargs.get('validation_ds', {})
        self.test_ds = kwargs.get('test_ds', {})
        
        # Optimizer parameters
        self.optim = kwargs.get('optim', {})
        self.scheduler = kwargs.get('scheduler', {})
        
        # Logging and checkpointing
        self.log_every_n_steps = kwargs.get('log_every_n_steps', 10)
        self.val_check_interval = kwargs.get('val_check_interval', 1.0)
        self.save_top_k = kwargs.get('save_top_k', 3)
        self.monitor = kwargs.get('monitor', 'val_loss')


class ModularModelNeMo(NeuralModule, Exportable):
    """
    NeMo wrapper for the ModularModel.
    
    This module provides full NeMo integration including:
    - Proper input/output type definitions
    - NeMo-compatible forward pass
    - Export capabilities
    - Integration with NeMo's training framework
    """
    
    def __init__(self, cfg: ModularModelConfig):
        super().__init__()
        
        self.cfg = cfg
        
        # Create decoder config object
        from types import SimpleNamespace
        decoder_config = SimpleNamespace(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_hidden_layers=cfg.num_layers,  # Add this for compatibility
            num_attention_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_kv_heads if cfg.num_kv_heads is not None else cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
            max_position_embeddings=cfg.max_position_embeddings,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            hidden_dropout_prob=cfg.hidden_dropout_prob,
            layer_norm_epsilon=cfg.layer_norm_epsilon,
            rms_norm_eps=cfg.rms_norm_eps,
            activation=cfg.activation,
            use_cache=cfg.use_cache,
            pad_token_id=cfg.pad_token_id,
            bos_token_id=cfg.bos_token_id,
            eos_token_id=cfg.eos_token_id,
            attention_type=cfg.attention_type,
            mlp_type=cfg.mlp_type,
            use_flash_attention=cfg.use_flash_attention,
            rotary_base=cfg.rotary_base,
        )
        
        # Create model config dictionary
        model_config = {
            'decoder_config': decoder_config,
            'vocab_size': cfg.vocab_size,
            # 'pool_type': cfg.pool_type,  # Not needed for Stage 1
            # 'num_reasoning_vectors': cfg.num_reasoning_vectors,  # Not needed for Stage 1
        }
        
        # Initialize the core model based on training stage
        if cfg.training_stage == 'stage1':
            # Stage 1: Use DecoderOnlyModel for memory efficiency
            self.model = DecoderOnlyModel(decoder_config, cfg.vocab_size, tie_weights=cfg.tie_weights)
            self.model_type = 'decoder_only'
        elif cfg.training_stage == 'stage2':
            # Stage 2: Use full ModularModel
            self.model = ModularModel(
                config=model_config,
                embedder_checkpoint_path=cfg.embedder_checkpoint_path,
                freeze_embedder_decoder=cfg.freeze_embedder_decoder,
                tie_weights=cfg.tie_weights,
                num_reasoning_vectors=cfg.num_reasoning_vectors
            )
            self.model_type = 'modular'
        else:
            raise ValueError(f"Unknown training stage: {cfg.training_stage}")
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)
        
        # Logging
        self.logger = nemo_logging
        
        # Training stage tracking
        self.current_stage = cfg.training_stage
        
        self.logger.info(f"Initialized ModularModelNeMo in {self.current_stage} mode with {self.model_type} model")
    
    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Define input types for NeMo type checking."""
        if self.current_stage == "stage1":
            return {
                "input_ids": NeuralType(('B', 'T'), ChannelType()),
                "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
                "labels": NeuralType(('B', 'T'), ChannelType(), optional=True),
                "target_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            }
        else:  # stage2
            return {
                "input_ids": NeuralType(('B', 'T'), ChannelType()),
                "embed_input_ids": NeuralType(('B', 'T'), ChannelType()),
                "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
                "embed_attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
                "labels": NeuralType(('B', 'T'), ChannelType(), optional=True),
                "target_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            }
    
    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Define output types for NeMo type checking."""
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
            "loss": NeuralType(elements_type=LossType(), optional=True),
        }
    
    @typecheck()
    def forward(
        self,
        # Stage 1 inputs
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # Stage 2 inputs
        embed_input_ids: Optional[torch.Tensor] = None,
        embed_attention_mask: Optional[torch.Tensor] = None,
        # Common inputs
        labels: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with NeMo-compatible interface.
        
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Forward pass through the core model
        if self.current_stage == "stage1":
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:  # stage2
            logits = self.model(
                input_ids=input_ids,
                embed_input_ids=embed_input_ids,
                attention_mask=attention_mask,
                embed_attention_mask=embed_attention_mask,
            )
        
        # Calculate loss if labels or target_ids are provided
        loss = None
        if target_ids is not None:
            # Use target_ids directly (already shifted for next token prediction)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        elif labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            "logits": logits,
            "loss": loss,
        }
    
    def set_training_stage(self, stage: str):
        """Change training stage dynamically."""
        if stage not in ["stage1", "stage2"]:
            raise ValueError(f"Invalid training stage: {stage}. Must be 'stage1' or 'stage2'")
        
        self.current_stage = stage
        
        # Create decoder config object
        from types import SimpleNamespace
        decoder_config = SimpleNamespace(
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            num_hidden_layers=self.cfg.num_layers,  # Add this for compatibility
            num_attention_heads=self.cfg.num_attention_heads,
            num_kv_heads=self.cfg.num_kv_heads if self.cfg.num_kv_heads is not None else self.cfg.num_attention_heads,
            intermediate_size=self.cfg.intermediate_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            dropout=self.cfg.dropout,
            attention_dropout=self.cfg.attention_dropout,
            hidden_dropout_prob=self.cfg.hidden_dropout_prob,
            layer_norm_epsilon=self.cfg.layer_norm_epsilon,
            rms_norm_eps=self.cfg.rms_norm_eps,
            activation=self.cfg.activation,
            use_cache=self.cfg.use_cache,
            pad_token_id=self.cfg.pad_token_id,
            bos_token_id=self.cfg.bos_token_id,
            eos_token_id=self.cfg.eos_token_id,
            attention_type=self.cfg.attention_type,
            mlp_type=self.cfg.mlp_type,
            use_flash_attention=self.cfg.use_flash_attention,
            rotary_base=self.cfg.rotary_base,
        )
        
        # Recreate model for new stage
        if stage == 'stage1' and self.model_type != 'decoder_only':
            # Switch to DecoderOnlyModel
            self.model = DecoderOnlyModel(decoder_config, self.cfg.vocab_size, tie_weights=self.cfg.tie_weights)
            self.model_type = 'decoder_only'
            
        elif stage == 'stage2' and self.model_type != 'modular':
            # Switch to ModularModel
            model_config = {
                'decoder_config': decoder_config,
                'vocab_size': self.cfg.vocab_size,
                'pool_type': self.cfg.pool_type,
                'num_reasoning_vectors': self.cfg.num_reasoning_vectors,
            }
            self.model = ModularModel(
                config=model_config,
                embedder_checkpoint_path=self.cfg.embedder_checkpoint_path,
                freeze_embedder_decoder=self.cfg.freeze_embedder_decoder,
                tie_weights=self.cfg.tie_weights,
                num_reasoning_vectors=self.cfg.num_reasoning_vectors
            )
            self.model_type = 'modular'
        
        self.logger.info(f"Switched to training stage: {stage} with {self.model_type} model")
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters based on current training stage."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        embed_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embed_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        if eos_token_id is None:
            eos_token_id = self.cfg.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.cfg.pad_token_id
        
        # Handle different model types
        if self.model_type == 'decoder_only':
            # DecoderOnlyModel doesn't need embed_input_ids
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        else:
            # ModularModel needs embed_input_ids for stage 2
            return self.model.generate(
                input_ids=input_ids,
                embed_input_ids=embed_input_ids,
                attention_mask=attention_mask,
                embed_attention_mask=embed_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
    
    def export(
        self,
        output: str,
        input_example: Optional[Dict[str, torch.Tensor]] = None,
        output_example: Optional[Dict[str, torch.Tensor]] = None,
        verbose: bool = True,
        export_params: bool = True,
        do_constant_folding: bool = True,
        keep_initializers_as_inputs: bool = False,
        onnx_opset_version: int = 12,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = True,
        use_dynamic_axes: bool = True,
    ) -> str:
        """
        Export the model to ONNX format.
        
        Args:
            output: Output file path
            input_example: Example input for tracing
            output_example: Example output for verification
            verbose: Whether to print export information
            export_params: Whether to export parameters
            do_constant_folding: Whether to do constant folding
            keep_initializers_as_inputs: Whether to keep initializers as inputs
            onnx_opset_version: ONNX opset version
            try_script: Whether to try scripting
            set_eval: Whether to set model to eval mode
            check_trace: Whether to check trace
            use_dynamic_axes: Whether to use dynamic axes
            
        Returns:
            Path to exported model
        """
        if input_example is None:
            # Create default input example based on current stage
            if self.current_stage == "stage1":
                input_example = {
                    "input_ids": torch.randint(0, self.cfg.vocab_size, (1, 10)),
                    "attention_mask": torch.ones(1, 10),
                }
            else:
                input_example = {
                    "input_ids": torch.randint(0, self.cfg.vocab_size, (1, 10)),
                    "embed_input_ids": torch.randint(0, self.cfg.vocab_size, (1, 10)),
                    "attention_mask": torch.ones(1, 10),
                    "embed_attention_mask": torch.ones(1, 10),
                }
        
        # Set model to eval mode
        if set_eval:
            self.eval()
        
        # Export to ONNX
        torch.onnx.export(
            self,
            tuple(input_example.values()),
            output,
            export_params=export_params,
            opset_version=onnx_opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            verbose=verbose,
            input_names=list(input_example.keys()),
            output_names=["logits"],
            dynamic_axes={
                name: {0: "batch_size", 1: "sequence_length"}
                for name in input_example.keys()
            } if use_dynamic_axes else None,
        )
        
        if verbose:
            self.logger.info(f"Model exported to: {output}")
        
        return output
    
    def save_to(self, save_path: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'cfg': self.cfg,
            'current_stage': self.current_stage,
            'model_type': self.model_type,
        }, save_path)
        self.logger.info(f"Model saved to: {save_path}")
    
    def restore_from(self, restore_path: str):
        """Restore the model from a file."""
        checkpoint = torch.load(restore_path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'current_stage' in checkpoint:
            self.current_stage = checkpoint['current_stage']
        if 'model_type' in checkpoint:
            self.model_type = checkpoint['model_type']
        self.logger.info(f"Model restored from: {restore_path}")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        self.logger.info("Gradient checkpointing enabled")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        self.logger.info("Gradient checkpointing disabled")


class ModularModelNeMoWrapper(NeuralModule):
    """
    Main NeMo model class for the ModularModel.
    
    This class provides the complete NeMo integration including:
    - Model instantiation
    - Training configuration
    - Validation and testing
    - Checkpointing and restoration
    """
    
    def __init__(self, cfg: ModularModelConfig):
        super().__init__()
        
        self.cfg = cfg
        
        # Initialize the core module
        self.modular_model = ModularModelNeMo(cfg)
        
        # Logging
        self.logger = nemo_logging
        
        self.logger.info(f"Initialized ModularModelNeMoWrapper with config: {cfg}")
    
    def parameters(self, recurse: bool = True):
        """Return model parameters for PyTorch Lightning compatibility."""
        if hasattr(self.modular_model, 'model') and self.modular_model.model is not None:
            # Use the actual ModularModel parameters
            return self.modular_model.model.parameters(recurse=recurse)
        else:
            # Fallback to modular_model parameters
            return self.modular_model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        """Return named model parameters for PyTorch Lightning compatibility."""
        if hasattr(self.modular_model, 'model') and self.modular_model.model is not None:
            # Use the actual ModularModel parameters
            return self.modular_model.model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        else:
            # Fallback to modular_model parameters
            return self.modular_model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def children(self):
        """Return child modules for PyTorch Lightning compatibility."""
        if hasattr(self.modular_model, 'model') and self.modular_model.model is not None:
            # Use the actual ModularModel children
            return self.modular_model.model.children()
        else:
            # Fallback to modular_model children
            return self.modular_model.children()
    
    def named_children(self):
        """Return named child modules for PyTorch Lightning compatibility."""
        if hasattr(self.modular_model, 'model') and self.modular_model.model is not None:
            # Use the actual ModularModel children
            return self.modular_model.model.named_children()
        else:
            # Fallback to modular_model children
            return self.modular_model.named_children()
    
    def modules(self):
        """Return all modules for PyTorch Lightning compatibility."""
        if hasattr(self.modular_model, 'model') and self.modular_model.model is not None:
            # Use the actual ModularModel modules
            return self.modular_model.model.modules()
        else:
            # Fallback to modular_model modules
            return self.modular_model.modules()
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Return all named modules for PyTorch Lightning compatibility."""
        if hasattr(self.modular_model, 'model') and self.modular_model.model is not None:
            # Use the actual ModularModel modules
            return self.modular_model.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
        else:
            # Fallback to modular_model modules
            return self.modular_model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
    
    def train(self, mode: bool = True):
        """Set training mode for PyTorch Lightning compatibility."""
        super().train(mode)
        return self.modular_model.train(mode)
    
    def eval(self):
        """Set evaluation mode for PyTorch Lightning compatibility."""
        super().eval()
        return self.modular_model.eval()
    
    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Delegate to the modular model."""
        return self.modular_model.input_types
    
    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Delegate to the modular model."""
        return self.modular_model.output_types
    
    @typecheck()
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Delegate to the modular model."""
        return self.modular_model(**kwargs)
    
    def set_training_stage(self, stage: str):
        """Change training stage."""
        self.modular_model.set_training_stage(stage)
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters."""
        return self.modular_model.get_trainable_parameters()
    
    def generate(self, **kwargs) -> torch.Tensor:
        """Generate text."""
        return self.modular_model.generate(**kwargs)
    
    def export(self, **kwargs) -> str:
        """Export model."""
        return self.modular_model.export(**kwargs)
    
    def save_to(self, save_path: str):
        """Save model."""
        self.modular_model.save_to(save_path)
    
    def restore_from(self, restore_path: str):
        """Restore model."""
        self.modular_model.restore_from(restore_path)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.modular_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.modular_model.gradient_checkpointing_disable()


# Factory function for easy model creation
def create_modular_model_nemo(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_attention_heads: int = 12,
    num_kv_heads: Optional[int] = None,
    intermediate_size: int = 3072,
    max_position_embeddings: int = 512,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    hidden_dropout_prob: float = 0.1,
    layer_norm_epsilon: float = 1e-6,
    rms_norm_eps: float = 1e-6,
    activation: str = "gelu",
    use_cache: bool = True,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pool_type: str = "mean",
    num_reasoning_vectors: int = 8,
    tie_weights: bool = True,
    freeze_embedder_decoder: bool = True,
    embedder_checkpoint_path: Optional[str] = None,
    attention_type: str = "gqa",
    mlp_type: str = "mlp",
    use_flash_attention: bool = True,
    rotary_base: float = 10000.0,
    training_stage: str = "stage1",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    **kwargs
) -> ModularModelNeMoWrapper:
    """
    Factory function to create a ModularModel with NeMo integration.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads for GQA (None for standard attention)
        intermediate_size: Intermediate layer size
        max_position_embeddings: Maximum sequence length
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        hidden_dropout_prob: Hidden dropout probability
        layer_norm_epsilon: Layer norm epsilon
        rms_norm_eps: RMS norm epsilon
        activation: Activation function
        use_cache: Whether to use caching
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pool_type: Pooling type for embedder
        num_reasoning_vectors: Number of reasoning vectors
        tie_weights: Whether to tie embedding and output weights
        freeze_embedder_decoder: Whether to freeze embedder decoder
        embedder_checkpoint_path: Path to embedder checkpoint
        attention_type: Attention type ('gqa' or 'vanilla')
        mlp_type: MLP type ('mlp' or 'gated')
        use_flash_attention: Whether to use flash attention
        rotary_base: Rotary embedding base
        training_stage: Training stage ("stage1" or "stage2")
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        **kwargs: Additional configuration parameters
        
    Returns:
        ModularModelNeMoWrapper instance
    """
    cfg = ModularModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        dropout=dropout,
        attention_dropout=attention_dropout,
        hidden_dropout_prob=hidden_dropout_prob,
        layer_norm_epsilon=layer_norm_epsilon,
        rms_norm_eps=rms_norm_eps,
        activation=activation,
        use_cache=use_cache,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pool_type=pool_type,
        num_reasoning_vectors=num_reasoning_vectors,
        tie_weights=tie_weights,
        freeze_embedder_decoder=freeze_embedder_decoder,
        embedder_checkpoint_path=embedder_checkpoint_path,
        attention_type=attention_type,
        mlp_type=mlp_type,
        use_flash_attention=use_flash_attention,
        rotary_base=rotary_base,
        training_stage=training_stage,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        **kwargs
    )
    
    return ModularModelNeMoWrapper(cfg)


def create_modular_model_from_existing_config(model_config_key: str = "model_config_1.8B",
                                            stage: str = "stage1",
                                            base_path: Optional[str] = None) -> 'ModularModelNeMoWrapper':
    """
    Create a ModularModelNeMoWrapper from existing configuration files.
    
    Args:
        model_config_key: Key for the model configuration in config.json
        stage: Training stage (stage1, stage2, etc.)
        base_path: Base path to the src directory
        
    Returns:
        ModularModelNeMoWrapper instance
    """
    if ConfigLoader is None:
        raise ImportError("ConfigLoader not available. Make sure config_loader.py is accessible.")
    
    # Create configuration from existing files
    config_dict = create_nemo_config_from_existing(model_config_key, stage, base_path)
    
    # Convert to ModularModelConfig
    cfg = ModularModelConfig()
    for key, value in config_dict.items():
        setattr(cfg, key, value)
    
    return ModularModelNeMoWrapper(cfg)


# Example usage and testing
if __name__ == "__main__":
    # Create a model instance
    model = create_modular_model_nemo(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,  # GQA with 2 key-value heads
        intermediate_size=1024,
        max_position_embeddings=128,
        num_reasoning_vectors=4,
        training_stage="stage1",
        attention_type="gqa",
        mlp_type="gated",
        use_flash_attention=True
    )
    
    print(f"Model created successfully!")
    print(f"Training stage: {model.modular_model.current_stage}")
    print(f"Model type: {model.modular_model.model_type}")
    print(f"Trainable parameters: {len(model.get_trainable_parameters())}")
    
    # Test forward pass
    batch_size = 2
    seq_length = 10
    
    # Stage 1 test
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 1000, (batch_size, seq_length))
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Stage 1 outputs:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss']}")
    
    # Switch to stage 2
    model.set_training_stage("stage2")
    
    # Stage 2 test
    embed_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    embed_attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(
        input_ids=input_ids,
        embed_input_ids=embed_input_ids,
        attention_mask=attention_mask,
        embed_attention_mask=embed_attention_mask,
        labels=labels
    )
    
    print(f"\nStage 2 outputs:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss']}")
    
    print(f"\nNeMo integration successful! 🎉")


def create_modular_model_nemo(
    vocab_size: int = 32000,
    hidden_size: int = 2048,
    num_layers: int = 24,
    num_attention_heads: int = 16,
    intermediate_size: int = 8192,
    hidden_dropout_prob: float = 0.1,
    training_stage: str = "stage1",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    **kwargs
) -> ModularModelNeMoWrapper:
    """
    Factory function to create a ModularModel with NeMo integration.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Feed-forward intermediate size
        hidden_dropout_prob: Dropout probability
        training_stage: Training stage (stage1, stage2)
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        **kwargs: Additional configuration parameters
        
    Returns:
        ModularModelNeMoWrapper instance
    """
    cfg = ModularModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        training_stage=training_stage,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        **kwargs
    )
    
    return ModularModelNeMoWrapper(cfg)


class ModularModelConfig:
    """Configuration class for ModularModel."""
    
    def __init__(self, **kwargs):
        # Model architecture parameters (Stage 1 - DecoderOnlyModel)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.hidden_size = kwargs.get('hidden_size', 2048)
        self.num_layers = kwargs.get('num_layers', 24)
        self.num_attention_heads = kwargs.get('num_attention_heads', 16)
        self.num_kv_heads = kwargs.get('num_kv_heads', None)  # For GQA support
        self.intermediate_size = kwargs.get('intermediate_size', 8192)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        
        # Additional model parameters for DecoderOnlyModel compatibility
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 4096)
        self.dropout = kwargs.get('dropout', 0.1)
        self.attention_dropout = kwargs.get('attention_dropout', 0.1)
        self.layer_norm_epsilon = kwargs.get('layer_norm_epsilon', 1e-6)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)  # For RMSNorm
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.use_cache = kwargs.get('use_cache', True)
        self.tie_weights = kwargs.get('tie_weights', True)
        self.activation = kwargs.get('activation', 'silu')  # For MLP activation
        self.rotary_base = kwargs.get('rotary_base', 10000.0)  # For RoPE
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.attention_type = kwargs.get('attention_type', 'gqa')
        self.mlp_type = kwargs.get('mlp_type', 'mlp')
        
        # Token-related parameters
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.eos_token_id = kwargs.get('eos_token_id', 2)
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        
        # Training parameters
        self.training_stage = kwargs.get('training_stage', 'stage1')
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.warmup_steps = kwargs.get('warmup_steps', 1000)
        
        # Additional parameters
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

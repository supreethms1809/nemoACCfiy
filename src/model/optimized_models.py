"""
Optimized model implementations for different training stages.

This module provides memory-efficient models that only load the components
needed for each specific training stage, avoiding the overhead of unused
components in the full ModularModel.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
try:
    from .embed_decoder_model import LMHeadDecoder
except ImportError:
    from embed_decoder_model import LMHeadDecoder


class DecoderOnlyModel(nn.Module):
    """
    Memory-optimized model for Stage 0 and Stage 1 training.
    
    Only contains the decoder component, avoiding the memory overhead
    of the unused embedder component in ModularModel.
    
    This reduces memory usage by ~50% for decoder-only training.
    """
    
    def __init__(self, config, vocab_size: int, tie_weights: bool = True):
        super().__init__()
        # Handle both dict and object config formats for compatibility
        if isinstance(config, dict):
            decoder_config = config['decoder_config']
            actual_vocab_size = config.get('vocab_size', vocab_size)
        else:
            decoder_config = config
            actual_vocab_size = vocab_size
            
        self.decoder = LMHeadDecoder(decoder_config, actual_vocab_size, tie_weights=tie_weights)
        self.hidden_size = decoder_config.hidden_size
        self.gradient_checkpointing = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        embed_input_ids: Optional[torch.Tensor] = None,  # Ignored for compatibility
        embed_attention_mask: Optional[torch.Tensor] = None,  # Ignored for compatibility
    ):
        """
        Forward pass for decoder-only training.
        
        Args:
            input_ids: Input token IDs (B, T)
            attention_mask: Attention mask (B, T)
            embed_input_ids: Ignored (for compatibility with ModularModel interface)
            embed_attention_mask: Ignored (for compatibility with ModularModel interface)
            
        Returns:
            logits: Output logits (B, T, V)
        """
        # Use gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                input_ids,
                attention_mask,
                use_reentrant=False
            )
        else:
            return self._forward_impl(input_ids, attention_mask)
    
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Implementation of forward pass."""
        # LMHeadDecoder expects input_ids and attention_mask as the first two parameters
        # and returns logits directly (no need for separate LM head)
        # Use minimal parameters to match what was working before
        logits = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,  # No cross-attention
            cross_attn_mask=None,
            is_causal=True,  # Decoder is always causal
        )
        return logits
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, temperature=1.0, 
                 top_p=1.0, top_k=50, do_sample=True, eos_token_id=None, pad_token_id=None):
        """Generate text using the decoder-only model."""
        # Delegate to the decoder's generate method
        return self.decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on DecoderOnlyModel")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.decoder, 'gradient_checkpointing_disable'):
            self.decoder.gradient_checkpointing_disable()
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


class EmbedderOnlyModel(nn.Module):
    """
    Memory-optimized model for embedder-only training.
    
    Only contains the embedder component, avoiding the memory overhead
    of the unused decoder component.
    """
    
    def __init__(self, config, vocab_size: int, pool_type: str = 'mean'):
        super().__init__()
        try:
            from .embed_decoder_model import Embedder
        except ImportError:
            from embed_decoder_model import Embedder
        # Handle both dict and object config formats for compatibility
        if isinstance(config, dict):
            decoder_config = config['decoder_config']
            actual_vocab_size = config.get('vocab_size', vocab_size)
            actual_pool_type = config.get('pool_type', pool_type)
        else:
            decoder_config = config
            actual_vocab_size = vocab_size
            actual_pool_type = pool_type
            
        self.embedder = Embedder(decoder_config, actual_vocab_size, actual_pool_type)
        self.hidden_size = decoder_config.hidden_size
        self.gradient_checkpointing = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for embedder-only training.
        
        Args:
            input_ids: Input token IDs (B, T)
            attention_mask: Attention mask (B, T)
            
        Returns:
            embeddings: Output embeddings (B, T, H) - N vectors per token
        """
        # Use gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                input_ids,
                attention_mask,
                use_reentrant=False
            )
        else:
            return self._forward_impl(input_ids, attention_mask)
    
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Implementation of forward pass."""
        embeddings = self.embedder(input_ids, attention_mask)
        return embeddings
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if hasattr(self.embedder, 'gradient_checkpointing_enable'):
            self.embedder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on EmbedderOnlyModel")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.embedder, 'gradient_checkpointing_disable'):
            self.embedder.gradient_checkpointing_disable()
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


def create_optimized_model(model_type: str, config: Dict[str, Any], **kwargs) -> nn.Module:
    """
    Factory function to create optimized models for different training stages.
    
    Args:
        model_type: Type of model to create ('decoder_only', 'embedder_only', 'modular')
        config: Model configuration dictionary
        **kwargs: Additional arguments for model creation
        
    Returns:
        nn.Module: Optimized model instance
    """
    if model_type == 'decoder_only':
        tie_weights = kwargs.get('tie_weights', True)
        return DecoderOnlyModel(config['decoder_config'], config['vocab_size'], tie_weights=tie_weights)
    elif model_type == 'embedder_only':
        pool_type = config.get('pool_type', 'mean')
        return EmbedderOnlyModel(config['decoder_config'], config['vocab_size'], pool_type)
    elif model_type == 'modular':
        try:
            from .embed_decoder_model import ModularModel
        except ImportError:
            from embed_decoder_model import ModularModel
        return ModularModel(config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_memory_usage_comparison():
    """
    Get memory usage comparison between different model types.
    
    Returns:
        dict: Memory usage comparison information
    """
    return {
        'decoder_only': {
            'description': 'Only decoder component',
            'memory_usage': '~50% of ModularModel',
            'use_cases': ['Stage 0 training', 'Stage 1 training'],
            'components': ['decoder']
        },
        'embedder_only': {
            'description': 'Only embedder component',
            'memory_usage': '~50% of ModularModel',
            'use_cases': ['Embedder pre-training'],
            'components': ['embedder']
        },
        'modular': {
            'description': 'Full model with both components',
            'memory_usage': '100% (baseline)',
            'use_cases': ['Stage 2 training', 'Inference'],
            'components': ['embedder', 'decoder']
        }
    }

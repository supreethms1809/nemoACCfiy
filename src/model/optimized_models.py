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
    
    def __init__(self, config, vocab_size: int, tie_weights: bool = False):
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
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Generate text using HuggingFace's generation utilities.
        
        This method uses HuggingFace's GenerationMixin.generate() which provides
        optimized generation with support for various decoding strategies, beam search,
        and advanced sampling techniques.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            repetition_penalty: Penalty for repetition (1.0 = no penalty, >1.0 = penalty)
            length_penalty: Length penalty for beam search
            early_stopping: Whether to stop early when EOS is generated
            use_cache: Whether to use KV cache for efficiency
            **kwargs: Additional arguments passed to HuggingFace's generate() method
            
        Returns:
            Generated token IDs of shape (batch_size, original_seq_len + new_tokens)
        """
        # Set model to eval mode
        self.eval()
        
        # Prepare generation kwargs for HuggingFace's generate method
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "top_k": top_k if do_sample else None,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty if repetition_penalty != 1.0 else None,
            "use_cache": use_cache,
            **kwargs
        }
        
        # Remove None values to use defaults
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        # Set token IDs if provided
        # IMPORTANT: Set pad_token_id first, then eos_token_id
        # Make sure pad_token_id is not the same as eos_token_id to avoid generation issues
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        elif hasattr(self.decoder.config, 'pad_token_id') and self.decoder.config.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.decoder.config.pad_token_id
            
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            # Also update the decoder's generation_config to ensure consistency
            if hasattr(self.decoder, 'generation_config'):
                self.decoder.generation_config.eos_token_id = eos_token_id
        elif hasattr(self.decoder.config, 'eos_token_id') and self.decoder.config.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.decoder.config.eos_token_id
            # Also update the decoder's generation_config
            if hasattr(self.decoder, 'generation_config'):
                self.decoder.generation_config.eos_token_id = self.decoder.config.eos_token_id
        
        # Ensure pad_token_id is not the same as eos_token_id
        if generation_kwargs.get("pad_token_id") == generation_kwargs.get("eos_token_id"):
            # If they're the same, use a different pad token (e.g., unk_token_id or 0)
            if hasattr(self.decoder.config, 'unk_token_id') and self.decoder.config.unk_token_id is not None:
                generation_kwargs["pad_token_id"] = self.decoder.config.unk_token_id
            else:
                generation_kwargs["pad_token_id"] = 0  # Use 0 as fallback pad token
        
        # Call HuggingFace's generate method via the decoder (which inherits from GenerationMixin)
        # This provides optimized generation with caching, beam search, and more
        with torch.no_grad():
            generated_ids = self.decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=False,
                output_scores=False,
                **generation_kwargs
            )
        
        return generated_ids
    
    def generate_custom(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Custom generate function (non-HuggingFace) for comparison.
        
        This is a manual implementation of autoregressive generation with KV caching.
        It provides the same functionality as HuggingFace's generate() but with full
        control over the generation loop.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            pad_token_id: Token ID for padding (not used in custom generation, but kept for API compatibility)
            eos_token_id: Token ID for end of sequence
            repetition_penalty: Penalty for repetition (1.0 = no penalty, >1.0 = penalty)
            use_cache: Whether to use KV cache for efficiency
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Generated token IDs of shape (batch_size, original_seq_len + new_tokens)
        """
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)
        
        # Prepare attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        
        # Prime the cache by running the full prompt once
        with torch.no_grad():
            logits, past_key_values = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=use_cache,
                is_causal=True,
            )
        
        # Start with the input_ids
        generated = [input_ids]  # list of tensors to cat at the end
        cur_token = input_ids[:, -1:]  # last token of prompt
        
        # Track generated tokens for repetition penalty
        generated_tokens = input_ids.tolist() if repetition_penalty != 1.0 else None
        
        # Helper: sample one step with repetition penalty
        def sample_from_logits(step_logits, prev_tokens=None):
            # Apply repetition penalty if enabled
            if repetition_penalty != 1.0 and prev_tokens is not None:
                # Penalize tokens that have been generated before
                for token_id in prev_tokens:
                    if step_logits[0, token_id] > 0:
                        step_logits[0, token_id] /= repetition_penalty
                    else:
                        step_logits[0, token_id] *= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                step_logits = step_logits / max(1e-6, temperature)
            
            probs = torch.softmax(step_logits, dim=-1)
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                mask = torch.full_like(probs, 0.0)
                mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)
            
            # Apply top-p (nucleus) sampling
            if top_p is not None and 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > top_p).float()
                # Keep at least 1 token
                cutoff[..., 0] = 0.0
                sorted_probs = sorted_probs * (1.0 - cutoff)
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
                # Map back
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            
            if do_sample:
                return torch.multinomial(probs, num_samples=1)
            else:
                return torch.argmax(probs, dim=-1, keepdim=True)
        
        # Generate new tokens
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Only feed the last generated token this step
                step_input = cur_token  # (B, 1)
                step_mask = torch.ones((B, 1), dtype=torch.long, device=device)
                
                # Forward pass with cached key-values
                step_logits, past_key_values = self.decoder(
                    input_ids=step_input,
                    attention_mask=step_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    is_causal=True,
                )
                
                # Get logits for the last position
                step_logits = step_logits[:, -1, :]  # (B, V)
                
                # Sample next token
                prev_tokens = generated_tokens[0] if generated_tokens is not None else None
                next_token = sample_from_logits(step_logits, prev_tokens)
                
                # Append to generated sequence
                generated.append(next_token)
                cur_token = next_token
                
                # Update generated tokens for repetition penalty
                if generated_tokens is not None:
                    generated_tokens[0].append(next_token[0, 0].item())
                
                # Check for EOS token
                if eos_token_id is not None:
                    if torch.all(next_token.squeeze(-1) == eos_token_id):
                        break
        
        # Concatenate all generated tokens
        return torch.cat(generated, dim=1)  # (B, original_seq_len + new_tokens)
    
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
        tie_weights = kwargs.get('tie_weights', False)
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

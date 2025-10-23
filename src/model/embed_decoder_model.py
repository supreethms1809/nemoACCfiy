import math
import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Flash Attention support
try:
    from .flash_attention import FlashAttentionMixin, FlashAttentionConfig, flash_attention_forward
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    FlashAttentionMixin = object
    FlashAttentionConfig = None
    flash_attention_forward = None

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for transformer models (pairwise)."""
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, D/2)
        sin, cos = freqs.sin().to(dtype), freqs.cos().to(dtype)
        # (1,1,T,D/2) for broadcasting over (B,H,T,D/2)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def apply(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x of shape (B,H,T,D)."""
        B, H, T, D = x.shape
        x = x.view(B, H, T, D // 2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        # sin/cos: (1,1,T,D/2)
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.stack((rot1, rot2), dim=-1).view(B, H, T, D)
        return out

class GQAttention(nn.Module, FlashAttentionMixin):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.key_heads = getattr(config, "num_kv_heads", self.num_heads)
        assert self.num_heads % self.key_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # Projections. For GQA, K/V produce key_heads*head_dim
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.key_heads * self.head_dim, bias=False)
        self.v = nn.Linear(self.hidden_size, self.key_heads * self.head_dim, bias=False)
        self.o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, base=getattr(config, "rotary_base", 10000.0))
        self.q_norm = RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))
        
        # Flash Attention configuration
        self.use_flash_attention = getattr(config, "use_flash_attention", True)
        if FLASH_ATTENTION_AVAILABLE:
            self.flash_attn_config = FlashAttentionConfig(
                use_flash_attention=self.use_flash_attention,
                dropout=getattr(config, "attention_dropout", 0.0),  # Use attention_dropout consistently
                causal=True
            )

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, T, _ = x.size()
        x = x.view(B, T, n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B,H,T,D)
        return x

    def _merge_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, H, T, D = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ):
        """
        hidden_states: (B, Tq, hidden)
        attention_mask: (B, Tq) with 1 for keep, 0 for mask (applies to queries for shape, but masking is on keys)
        key_value_states: (B, Tk, hidden) for cross-attention; if None → self-attention
        key_value_mask: (B, Tk) with 1 for keep, 0 for mask
        is_causal: Whether to apply causal masking (default True for backward compatibility)
        """
        B, Tq, _ = hidden_states.size()
        self_attn = key_value_states is None
        if key_value_states is None:
            key_value_states = hidden_states
        _, Tk, _ = key_value_states.size()

        # Projections
        q = self.q(hidden_states)                               # (B,Tq,H*D)
        k = self.k(key_value_states)                            # (B,Tk,K*D)
        v = self.v(key_value_states)                            # (B,Tk,K*D)

        # Split to heads
        q = self._split_heads(q, self.num_heads)                # (B,H,Tq,D)
        k = k.view(B, Tk, self.key_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B,K,Tk,D)
        v = v.view(B, Tk, self.key_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B,K,Tk,D)

        # RoPE on q/k
        sin_q, cos_q = self.rotary_emb._build(Tq, q.device, q.dtype)
        q = self.q_norm(self.rotary_emb.apply(q, sin_q, cos_q))
        sin_k, cos_k = self.rotary_emb._build(Tk, k.device, k.dtype)
        k = self.k_norm(self.rotary_emb.apply(k, sin_k, cos_k))

        # Handle past_key_value for caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # Ensure both are (B, K, T, D)
            assert past_k.shape[1] == self.key_heads, f"past_k shape[1] {past_k.shape[1]} != key_heads {self.key_heads}"
            assert k.shape[1] == self.key_heads, f"k shape[1] {k.shape[1]} != key_heads {self.key_heads}"
            k = torch.cat([past_k, k], dim=2)  # Concatenate on sequence length
            v = torch.cat([past_v, v], dim=2)
            Tk = k.size(2)

        # Repeat kv heads if needed (GQA)
        k_for_attn = k
        v_for_attn = v
        if self.num_heads != self.key_heads:
            repeat_kv = self.num_heads // self.key_heads
            k_for_attn = k.repeat_interleave(repeat_kv, dim=1)  # (B,H,Tk,D)
            v_for_attn = v.repeat_interleave(repeat_kv, dim=1)

        # Use Flash Attention if available and enabled
        if self.use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            # Prepare attention mask for Flash Attention
            attn_mask = None
            if key_value_mask is None and attention_mask is not None and self_attn:
                key_value_mask = attention_mask
            
            if key_value_mask is not None:
                # Convert to (B, 1, Tq, Tk) format for Flash Attention
                attn_mask = key_value_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, Tq, -1)
                # Convert to boolean mask (True for keep, False for mask)
                # key_value_mask: 1=valid token, 0=padding token
                # Flash attention expects: True=keep, False=mask
                attn_mask = attn_mask.bool()
            
            # Use Flash Attention
            scale = 1.0 / math.sqrt(self.head_dim)
            context = self.flash_attention_forward(
                query=q,
                key=k_for_attn,
                value=v_for_attn,
                attention_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self_attn and is_causal,
                scale=scale
            )
        else:
            # Fallback to standard attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k_for_attn.transpose(-1, -2)) * scale  # (B,H,Tq,Tk)

            # Causal mask only for self-attention and when is_causal is True
            if self_attn and is_causal:
                causal = torch.full((1, 1, Tq, Tk), float('-inf'), device=hidden_states.device, dtype=attn_scores.dtype)
                causal = torch.triu(causal, diagonal=1)
                attn_scores = attn_scores + causal

            # Key padding mask (mask keys)
            if key_value_mask is None and attention_mask is not None and self_attn:
                key_value_mask = attention_mask
            if key_value_mask is not None:
                # (B,1,1,Tk) - mask positions where key_value_mask is 0 (padding tokens)
                mask = (key_value_mask.float() == 0.0).unsqueeze(1).unsqueeze(2).to(attn_scores.dtype) * -1e9
                attn_scores = attn_scores + mask

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = torch.matmul(attn_weights, v_for_attn)  # (B,H,Tq,D)

        context = self._merge_heads(context, self.num_heads)    # (B,Tq,H*D)
        output = self.o(context)                                 # (B,Tq,hidden)

        if use_cache:
            # Return output and new key/value for caching (always (B, K, T, D))
            return output, (k, v)
        else:
            return output

class VanillaAttention(nn.Module, FlashAttentionMixin):
    """Vanilla multi-head attention implementation with self and cross attention support."""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # Linear projections for Q, K, V, O
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Optional: Add bias to output projection
        self.o_bias = getattr(config, "attention_output_bias", False)
        if self.o_bias:
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, "attention_dropout", 0.0))
        
        # Optional: Add layer norm for Q, K
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=getattr(config, "layer_norm_eps", 1e-6))
            self.k_norm = nn.LayerNorm(self.head_dim, eps=getattr(config, "layer_norm_eps", 1e-6))
        
        # Flash Attention configuration
        self.use_flash_attention = getattr(config, "use_flash_attention", True)
        if FLASH_ATTENTION_AVAILABLE:
            self.flash_attn_config = FlashAttentionConfig(
                use_flash_attention=self.use_flash_attention,
                dropout=getattr(config, "attention_dropout", 0.0),
                causal=True
            )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()  # (batch, num_heads, seq_len, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the heads back into the last dimension."""
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, head_dim)
        return x.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - Query states
            attention_mask: (batch_size, seq_len) - Attention mask for queries
            key_value_states: (batch_size, kv_seq_len, hidden_size) - Key/Value states (for cross-attention)
            key_value_mask: (batch_size, kv_seq_len) - Attention mask for keys/values
            past_key_value: Tuple of (key, value) from previous forward pass
            use_cache: Whether to return key/value for caching
            is_causal: Whether to apply causal masking (default True for backward compatibility)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Determine if this is self-attention or cross-attention
        is_cross_attention = key_value_states is not None
        
        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
        else:
            key_value_states = hidden_states
            kv_seq_len = seq_len
            if key_value_mask is None and attention_mask is not None:
                key_value_mask = attention_mask

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)  # (batch, seq_len, hidden_size)
        key_states = self.k_proj(key_value_states)  # (batch, kv_seq_len, hidden_size)
        value_states = self.v_proj(key_value_states)  # (batch, kv_seq_len, hidden_size)

        # Split into heads
        query_states = self._split_heads(query_states)  # (batch, num_heads, seq_len, head_dim)
        key_states = self._split_heads(key_states)  # (batch, num_heads, kv_seq_len, head_dim)
        value_states = self._split_heads(value_states)  # (batch, num_heads, kv_seq_len, head_dim)

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Handle past key/value states for caching
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            kv_seq_len = key_states.size(2)

        # Use Flash Attention if available and enabled
        if self.use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            # Prepare attention mask for Flash Attention
            attn_mask = None
            if key_value_mask is not None:
                # Convert to (batch, 1, seq_len, kv_seq_len) format for Flash Attention
                attn_mask = key_value_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_len, -1)
                # Convert to boolean mask (True for keep, False for mask)
                attn_mask = attn_mask.bool()
            
            # Use Flash Attention
            scale = 1.0 / math.sqrt(self.head_dim)
            context_states = self.flash_attention_forward(
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=not is_cross_attention and is_causal,
                scale=scale
            )
        else:
            # Fallback to standard attention
            # Calculate attention scores
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.head_dim)

            # Apply attention masks
            if not is_cross_attention and is_causal:
                # Causal mask for self-attention only when is_causal is True
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_seq_len, device=hidden_states.device, dtype=torch.bool),
                    diagonal=1
                )
                attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Apply key/value padding mask
            if key_value_mask is not None:
                # Expand mask to match attention scores shape
                mask = key_value_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, kv_seq_len)
                # Mask positions where key_value_mask is 0 (padding tokens)
                attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

            # Apply softmax and dropout
            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Apply attention to values
            context_states = torch.matmul(attention_probs, value_states)  # (batch, num_heads, seq_len, head_dim)

        # Merge heads
        context_states = self._merge_heads(context_states)  # (batch, seq_len, hidden_size)

        # Output projection
        output = self.o_proj(context_states)

        # Return cached key/value if requested
        if use_cache:
            return output, (key_states, value_states)
        
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLPGated(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.hidden_size, config.intermediate_size)  # Separate path for value
        self.fc3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path: fc1 -> activation
        gate = self.activation(self.fc1(x))
        # Value path: fc2 (separate from gate)
        value = self.fc2(x)
        # Gated combination
        gated = gate * value
        # Final projection with dropout
        output = self.dropout(self.fc3(gated))
        return output

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Choose between GQAttention and VanillaAttention based on config
        attention_type = getattr(config, "attention_type", "gqa").lower()
        if attention_type == "vanilla" or attention_type == "standard":
            self.self_attn = VanillaAttention(config)
            self.cross_attn = VanillaAttention(config)
        else:
            self.self_attn = GQAttention(config)
            self.cross_attn = GQAttention(config)
    
        # Choose between MLP and MLPGated based on config
        mlp_type = getattr(config, "mlp_type", "mlp").lower()
        if mlp_type == "gated" or mlp_type == "mlp_gated":
            self.mlp = MLPGated(config)
        else:
            self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.norm2 = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.norm3 = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ):
        # Unpack past_key_values for self/cross attention if present
        if past_key_values is not None:
            if isinstance(past_key_values, tuple) and len(past_key_values) == 2:
                self_past, cross_past = past_key_values
            else:
                self_past, cross_past = None, None
        else:
            self_past, cross_past = None, None

        # Self-attention
        residual = x
        x = self.norm1(x)  # Apply layer norm before self-attention
        if use_cache:
            x, self_cache = self.self_attn(
                x,
                attention_mask=self_attn_mask,
                key_value_states=None,
                key_value_mask=None,
                past_key_value=self_past,
                use_cache=use_cache,
                is_causal=is_causal,
            )
        else:
            x = self.self_attn(
                x,
                attention_mask=self_attn_mask,
                key_value_states=None,
                key_value_mask=None,
                past_key_value=self_past,
                use_cache=use_cache,
                is_causal=is_causal,
            )
            self_cache = None
        x = residual + x

        # Cross-attention (if provided)
        if encoder_hidden_states is not None:
            residual = x
            x = self.norm2(x)  # Apply layer norm before cross-attention
            if use_cache:
                x, cross_cache = self.cross_attn(
                    x,
                    attention_mask=self_attn_mask,
                    key_value_states=encoder_hidden_states,
                    key_value_mask=cross_attn_mask,
                    past_key_value=cross_past,
                    use_cache=use_cache,
                    is_causal=False,  # Cross-attention is never causal
                )
            else:
                x = self.cross_attn(
                    x,
                    attention_mask=self_attn_mask,
                    key_value_states=encoder_hidden_states,
                    key_value_mask=cross_attn_mask,
                    past_key_value=cross_past,
                    use_cache=use_cache,
                    is_causal=False,  # Cross-attention is never causal
                )
                cross_cache = None
            x = residual + x
        else:
            cross_cache = None

        # MLP
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = residual + x

        if use_cache:
            return x, (self_cache, cross_cache)
        else:
            return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.gradient_checkpointing = False
        
        # Initialize weights properly
        self._init_weights_manually()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        print("Gradient checkpointing enabled on Decoder")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def _init_weights(self, module):
        """Initialize weights using proper scaling for transformer models."""
        # Skip modules that don't need standard initialization
        if isinstance(module, (RotaryEmbedding,)):
            return
            
        if isinstance(module, nn.Linear):
            # Use the initializer_range from config if available
            initializer_range = getattr(self.config, 'initializer_range', 0.02)
            # Initialize with normal distribution scaled by initializer_range
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            initializer_range = getattr(self.config, 'initializer_range', 0.02)
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _init_weights_manually(self):
        """Manually initialize weights to avoid apply() conflicts."""
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        
        # Initialize decoder layers
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    param.data.normal_(mean=0.0, std=initializer_range)
                elif 'bias' in name:
                    param.data.zero_()
        
        # Initialize final norm
        self.norm.weight.data.fill_(1.0)
        if hasattr(self.norm, 'bias') and self.norm.bias is not None:
            self.norm.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        is_causal: bool = True,
    ):
        caches = [] if use_cache else None
        
        # Use gradient checkpointing if enabled and not using cache
        if self.gradient_checkpointing and self.training and not use_cache:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            for i, layer in enumerate(self.layers):
                layer_past = past_key_values[i] if (use_cache and past_key_values is not None) else None
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    encoder_hidden_states,
                    self_attn_mask,
                    cross_attn_mask,
                    layer_past,
                    use_cache,
                    is_causal,
                    use_reentrant=False
                )
        else:
            # Original forward pass
            for i, layer in enumerate(self.layers):
                layer_past = past_key_values[i] if (use_cache and past_key_values is not None) else None
                if use_cache:
                    hidden_states, layer_cache = layer(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        self_attn_mask=self_attn_mask,
                        cross_attn_mask=cross_attn_mask,
                        past_key_values=layer_past,
                        use_cache=use_cache,
                        is_causal=is_causal,
                    )
                    caches.append(layer_cache)
                else:
                    hidden_states = layer(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        self_attn_mask=self_attn_mask,
                        cross_attn_mask=cross_attn_mask,
                        past_key_values=layer_past,
                        use_cache=use_cache,
                        is_causal=is_causal,
                    )
        
        # Apply final normalization
        output = self.norm(hidden_states)
        if use_cache:
            return output, tuple(caches)
        else:
            return output

class LMHeadDecoder(nn.Module):
    """A simple token-embedding + Decoder + LM head wrapper with weight tying."""
    def __init__(self, config, vocab_size: int, tie_weights: bool = True):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size)
        self.decoder = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.gradient_checkpointing = False
        
        # Initialize weights properly
        self._init_weights_manually()
        
        # Weight tying: tie LM head weights to embedding weights
        if tie_weights:
            self.tie_weights()
            print(f"🔗 Weight tying enabled: LM head tied to embedding weights ({vocab_size:,} params saved)")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on LMHeadDecoder")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.decoder, 'gradient_checkpointing_disable'):
            self.decoder.gradient_checkpointing_disable()
    
    def _init_weights(self, module):
        """Initialize weights using proper scaling for transformer models."""
        # Skip modules that don't need standard initialization
        if isinstance(module, (RotaryEmbedding,)):
            return
            
        if isinstance(module, nn.Linear):
            # Use the initializer_range from config if available
            initializer_range = getattr(self.config, 'initializer_range', 0.02)
            # Initialize with normal distribution scaled by initializer_range
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            initializer_range = getattr(self.config, 'initializer_range', 0.02)
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _init_weights_manually(self):
        """Manually initialize weights to avoid apply() conflicts."""
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        
        # Initialize embedding weights
        self.embed_tokens.weight.data.normal_(mean=0.0, std=initializer_range)
        
        # Initialize LM head weights (will be tied to embeddings later)
        self.lm_head.weight.data.normal_(mean=0.0, std=initializer_range)
        
        # Initialize decoder layers
        for layer in self.decoder.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    param.data.normal_(mean=0.0, std=initializer_range)
                elif 'bias' in name:
                    param.data.zero_()
        
        # Initialize final norm
        self.decoder.norm.weight.data.fill_(1.0)
        if hasattr(self.decoder.norm, 'bias') and self.decoder.norm.bias is not None:
            self.decoder.norm.bias.data.zero_()
    
    def tie_weights(self):
        """Tie the weights of the LM head to the embedding weights."""
        self.lm_head.weight = self.embed_tokens.weight
        print("✅ LM head weights tied to embedding weights")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
        is_causal: bool = True,
    ):
        hidden = self.embed_tokens(input_ids)
        
        # Use gradient checkpointing if enabled and not using cache
        if self.gradient_checkpointing and self.training and not use_cache and not return_hidden_states:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.decoder),
                hidden,
                encoder_hidden_states,
                attention_mask,
                cross_attn_mask,
                past_key_values,
                use_cache,
                is_causal,
                use_reentrant=False
            )
            logits = self.lm_head(hidden)
            return logits
        
        # Original forward pass
        if use_cache:
            hidden, new_past = self.decoder(
                hidden_states=hidden,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=attention_mask,
                cross_attn_mask=cross_attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                is_causal=is_causal,
            )
        else:
            hidden = self.decoder(
                hidden_states=hidden,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=attention_mask,
                cross_attn_mask=cross_attn_mask,
                past_key_values=past_key_values,
                use_cache=False,
                is_causal=is_causal,
            )
            new_past = None
        logits = self.lm_head(hidden)
        if return_hidden_states:
            if use_cache:
                return logits, hidden, new_past
            else:
                return logits, hidden
        if use_cache:
            return logits, new_past
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
    ):
        """
        Generate text using the decoder-only model.
        
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
            
        Returns:
            Generated token IDs of shape (batch_size, original_seq_len + new_tokens)
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Get logits for the last token
            with torch.no_grad():
                if use_cache and past_key_values is not None:
                    # Use cache for efficiency - only process the last token
                    last_token_ids = generated_ids[:, -1:]
                    last_attention_mask = torch.ones_like(last_token_ids, dtype=torch.bool)
                    
                    logits, past_key_values = self.forward(
                        input_ids=last_token_ids,
                        attention_mask=last_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        is_causal=True,
                    )
                else:
                    # Process the entire sequence
                    logits, past_key_values = self.forward(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        is_causal=True,
                    )
                
                # Get logits for the last position
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for batch_idx in range(batch_size):
                        for token_id in generated_ids[batch_idx]:
                            if next_token_logits[batch_idx, token_id] < 0:
                                next_token_logits[batch_idx, token_id] *= repetition_penalty
                            else:
                                next_token_logits[batch_idx, token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or select the next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the new token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Update attention mask
                new_attention_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
                attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
                
                # Check for early stopping
                if early_stopping and eos_token_id is not None:
                    if (next_token == eos_token_id).all():
                        break
        
        return generated_ids

class LatentAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, mask=None):
        # hidden_states: (batch, seq_len, hidden)
        scores = self.attn(hidden_states).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden)
        return pooled

class LatentPooling(nn.Module):
    def __init__(self, pool_type='mean', hidden_size=None):
        super().__init__()
        self.pool_type = pool_type
        if self.pool_type == 'attention':
            if hidden_size is None:
                raise ValueError("hidden_size must be specified for attention pooling")
            self.attn_pool = LatentAttentionPooling(hidden_size)

    def forward(self, hidden_states, mask=None):
        if self.pool_type == 'mean':
            if mask is not None:
                hidden_states = hidden_states * mask.unsqueeze(-1)
                return hidden_states.sum(1) / mask.sum(1, keepdim=True)
            else:
                return hidden_states.mean(1)
        elif self.pool_type == 'max':
            return hidden_states.max(1)[0]
        elif self.pool_type == 'attention':
            return self.attn_pool(hidden_states, mask)
        else:
            raise NotImplementedError("Supported: mean, max, attention")

class LearnedQueryPooler(nn.Module):
    """
    Minimal "learned-query pooler" that uses learned queries to attend to specific token patterns.
    
    This replaces simple pooling with cross-attention between learned query vectors and token representations,
    allowing each reasoning vector to focus on different aspects of the input text.
    """
    def __init__(self, hidden_size: int, num_slots: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        
        # Learned query vectors - each represents a different reasoning aspect
        self.queries = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)
        
        # Projection layers for Q, K, V
        self.proj_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, token_h, attn_mask=None):
        """
        Args:
            token_h: Token hidden states [B, T, H]
            attn_mask: Attention mask [B, T] (True for valid tokens, False for padding)
        Returns:
            slots: Reasoning vectors [B, N, H]
        """
        B, T, H = token_h.shape
        
        # Project learned queries to query space
        Q = self.proj_q(self.queries).unsqueeze(0).expand(B, -1, -1)  # [B, N, H]
        
        # Project token hidden states to key and value spaces
        K = self.proj_k(token_h)  # [B, T, H]
        V = self.proj_v(token_h)  # [B, T, H]

        # Scaled dot-product attention
        scores = (Q @ K.transpose(1, 2)) / (H ** 0.5)  # [B, N, T]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # attn_mask: [B, T] where True = valid token, False = padding
            # We need to mask out padding tokens (False values)
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
        
        # Compute attention weights and apply to values
        attn = scores.softmax(dim=-1)  # [B, N, T]
        slots = attn @ V  # [B, N, H]
        
        return slots

class Embedder(nn.Module):
    """
    Embedder that processes tree-structured reasoning/planning text and extracts N reasoning vectors.
    
    Instead of returning token-level vectors, it extracts N meaningful reasoning components
    from the structured planning tree text.
    """
    def __init__(self, decoder_config, vocab_size, pool_type='learned_query', tie_weights=True, num_reasoning_vectors=8):
        super().__init__()
        self.decoder_model = LMHeadDecoder(decoder_config, vocab_size, tie_weights=tie_weights)
        hidden_size = decoder_config.hidden_size
        self.num_reasoning_vectors = num_reasoning_vectors
        self.gradient_checkpointing = False
        
        # Use LearnedQueryPooler for better reasoning vector extraction
        if pool_type == 'learned_query':
            self.pooler = LearnedQueryPooler(hidden_size, num_reasoning_vectors)
        else:
            # Fallback to original pooling methods
            self.pooler = LatentPooling(pool_type, hidden_size=hidden_size)
            # Keep the old reasoning extractor for non-learned_query pooling
            self.reasoning_extractor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * num_reasoning_vectors),
                nn.Tanh()  # Normalize reasoning vectors
            )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if hasattr(self.decoder_model, 'gradient_checkpointing_enable'):
            self.decoder_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on Embedder")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.decoder_model, 'gradient_checkpointing_disable'):
            self.decoder_model.gradient_checkpointing_disable()

    def forward(self, embed_input_ids, embed_attention_mask=None):
        # Disable gradient checkpointing for embedder to avoid return format issues
        # The embedder is relatively small compared to the main decoder
        result = self.decoder_model(
            input_ids=embed_input_ids,
            attention_mask=embed_attention_mask,  # Now properly use attention mask
            return_hidden_states=True,
            is_causal=False,  # Embedder uses bidirectional attention (full context)
        )
        
        # Handle different return formats
        if isinstance(result, tuple):
            if len(result) == 3:
                _logits, hidden, _ = result
            elif len(result) == 2:
                _logits, hidden = result
            else:
                raise ValueError(f"Unexpected tuple length: {len(result)}")
        else:
            raise ValueError(f"Expected tuple return, got {type(result)}")
        
        # Extract N reasoning vectors using the appropriate pooling method
        if isinstance(self.pooler, LearnedQueryPooler):
            # Use LearnedQueryPooler for better reasoning vector extraction
            reasoning_vectors = self.pooler(hidden, embed_attention_mask)  # (B, N, H)
        else:
            # Fallback to original approach for other pooling methods
            pooled_hidden = self.pooler(hidden, mask=embed_attention_mask)  # (B, H)
            reasoning_vectors = self.reasoning_extractor(pooled_hidden)  # (B, H * N)
            reasoning_vectors = reasoning_vectors.view(
                reasoning_vectors.size(0), 
                self.num_reasoning_vectors, 
                -1
            )  # (B, N, H)
        
        return reasoning_vectors  # N reasoning vectors per batch

class ModularModel(nn.Module):
    """
    Combines an Embedder and a Decoder with **always-on cross-attention**.
    The embedder's decoder can be loaded from a checkpoint and frozen.
    """
    def __init__(self, config, embedder_checkpoint_path=None, freeze_embedder_decoder=True, tie_weights=True, num_reasoning_vectors=8):
        super().__init__()
        self.embedder = Embedder(
            config['decoder_config'], 
            config['vocab_size'], 
            config.get('pool_type', 'mean'), 
            tie_weights=tie_weights,
            num_reasoning_vectors=num_reasoning_vectors
        )
        self.decoder = LMHeadDecoder(config['decoder_config'], config['vocab_size'], tie_weights=tie_weights)
        self.hidden_size = config['decoder_config'].hidden_size
        self.num_reasoning_vectors = num_reasoning_vectors
        self.gradient_checkpointing = False
        
        # Load embedder decoder from checkpoint if provided
        if embedder_checkpoint_path:
            self._load_embedder_from_checkpoint(embedder_checkpoint_path, freeze_embedder_decoder)
        else:
            # If no checkpoint provided but freezing is requested, freeze the randomly initialized embedder
            if freeze_embedder_decoder:
                self._freeze_embedder_decoder()
                print("🔒 Embedder decoder parameters frozen (no checkpoint loaded)")

    def _load_embedder_from_checkpoint(self, checkpoint_path, freeze_embedder_decoder=True):
        """
        Load the embedder's decoder from a checkpoint and optionally freeze its parameters.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            freeze_embedder_decoder (bool): Whether to freeze the embedder decoder parameters
        """
        import torch
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Embedder checkpoint not found: {checkpoint_path}")
        
        print(f"Loading embedder decoder from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # Load the decoder part into the embedder's decoder
        # The embedder's decoder should have the same structure as the checkpoint model
        embedder_decoder_state_dict = {}
        
        # Extract decoder parameters from the checkpoint
        # Assuming the checkpoint contains a decoder with the same structure
        for key, value in model_state_dict.items():
            # Map checkpoint keys to embedder decoder keys
            if key.startswith('decoder.') or key.startswith('model.'):
                # Remove the prefix to get the actual parameter name
                if key.startswith('decoder.'):
                    new_key = key[8:]  # Remove 'decoder.' prefix
                elif key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                else:
                    new_key = key
                
                embedder_decoder_state_dict[new_key] = value
            elif not key.startswith('embedder.'):
                # If it's not an embedder-specific key, it might be a decoder parameter
                embedder_decoder_state_dict[key] = value
        
        # Load the state dict into the embedder's decoder
        try:
            self.embedder.decoder_model.load_state_dict(embedder_decoder_state_dict, strict=False)
            print("✅ Successfully loaded embedder decoder from checkpoint")
        except Exception as e:
            print(f"⚠️ Warning: Could not load embedder decoder state dict: {e}")
            print("Continuing with randomly initialized embedder decoder...")
            return
        
        # Freeze embedder decoder parameters if requested
        if freeze_embedder_decoder:
            self._freeze_embedder_decoder()
            print("🔒 Embedder decoder parameters frozen")
        else:
            print("🔓 Embedder decoder parameters remain trainable")

    def _freeze_embedder_decoder(self):
        """Freeze only the embedder's decoder parameters, keep pooler trainable."""
        embedder_decoder_params = 0
        embedder_pooler_params = 0
        
        # Freeze only the decoder part
        for param in self.embedder.decoder_model.parameters():
            param.requires_grad = False
            embedder_decoder_params += 1
        
        # Keep the pooler trainable (don't freeze it)
        for param in self.embedder.pooler.parameters():
            param.requires_grad = True  # Keep pooler trainable
            embedder_pooler_params += 1
        
        print(f"🔒 Frozen {embedder_decoder_params} embedder decoder parameters")
        print(f"🔓 Kept {embedder_pooler_params} pooler parameters trainable")

    def _unfreeze_embedder_decoder(self):
        """Unfreeze all parameters in the embedder's decoder and pooler."""
        for param in self.embedder.decoder_model.parameters():
            param.requires_grad = True
        
        # Also unfreeze the embedder's pooler
        for param in self.embedder.pooler.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Get the number of trainable parameters, excluding frozen embedder decoder."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Account for weight tying savings
        weight_tying_savings = 0
        if hasattr(self.decoder, 'lm_head') and hasattr(self.decoder, 'embed_tokens'):
            if self.decoder.lm_head.weight is self.decoder.embed_tokens.weight:
                weight_tying_savings += self.decoder.embed_tokens.weight.numel()
        
        if hasattr(self.embedder, 'decoder_model'):
            if (hasattr(self.embedder.decoder_model, 'lm_head') and 
                hasattr(self.embedder.decoder_model, 'embed_tokens')):
                if self.embedder.decoder_model.lm_head.weight is self.embedder.decoder_model.embed_tokens.weight:
                    weight_tying_savings += self.embedder.decoder_model.embed_tokens.weight.numel()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'weight_tying_savings': weight_tying_savings,
            'effective_total': total_params + weight_tying_savings  # What it would be without tying
        }

    def verify_parameter_states(self):
        """Verify that embedder decoder parameters are frozen while pooler and main decoder remain trainable."""
        # Embedder components
        embedder_decoder_trainable = sum(p.numel() for p in self.embedder.decoder_model.parameters() if p.requires_grad)
        embedder_decoder_frozen = sum(p.numel() for p in self.embedder.decoder_model.parameters() if not p.requires_grad)
        embedder_pooler_trainable = sum(p.numel() for p in self.embedder.pooler.parameters() if p.requires_grad)
        embedder_pooler_frozen = sum(p.numel() for p in self.embedder.pooler.parameters() if not p.requires_grad)
        
        # Main decoder
        decoder_trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        decoder_frozen = sum(p.numel() for p in self.decoder.parameters() if not p.requires_grad)
        
        print("Parameter State Verification:")
        print(f"  Embedder Decoder - Trainable: {embedder_decoder_trainable:,}, Frozen: {embedder_decoder_frozen:,}")
        print(f"  Embedder Pooler  - Trainable: {embedder_pooler_trainable:,}, Frozen: {embedder_pooler_frozen:,}")
        print(f"  Main Decoder     - Trainable: {decoder_trainable:,}, Frozen: {decoder_frozen:,}")
        
        if embedder_decoder_frozen > 0:
            print("  ✅ Embedder decoder parameters are frozen")
        else:
            print("  ⚠️  Embedder decoder parameters are trainable")
            
        if embedder_pooler_trainable > 0:
            print("  ✅ Embedder pooler parameters are trainable")
        else:
            print("  ❌ Embedder pooler parameters are frozen (unexpected!)")
            
        if decoder_trainable > 0:
            print("  ✅ Main decoder parameters are trainable")
        else:
            print("  ❌ Main decoder parameters are frozen (unexpected!)")
        
        return {
            'embedder_decoder_trainable': embedder_decoder_trainable,
            'embedder_decoder_frozen': embedder_decoder_frozen,
            'embedder_pooler_trainable': embedder_pooler_trainable,
            'embedder_pooler_frozen': embedder_pooler_frozen,
            'decoder_trainable': decoder_trainable,
            'decoder_frozen': decoder_frozen
        }

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        # Enable on decoder only (not embedder to avoid return format issues)
        if hasattr(self.decoder, 'gradient_checkpointing_enable'):
            self.decoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on ModularModel (decoder only)")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        # Disable on submodules
        if hasattr(self.decoder, 'gradient_checkpointing_disable'):
            self.decoder.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids,
        embed_input_ids,
        attention_mask=None,
        embed_attention_mask=None,
    ):
        # Use gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                input_ids,
                embed_input_ids,
                attention_mask,
                embed_attention_mask,
                use_reentrant=False
            )
        else:
            return self._forward_impl(input_ids, embed_input_ids, attention_mask, embed_attention_mask)

    def _forward_impl(
        self,
        input_ids,
        embed_input_ids,
        attention_mask=None,
        embed_attention_mask=None,
    ):
        # Handle stage 1 training (decoder-only, no cross-attention)
        if embed_input_ids is None:
            # Stage 1: Only use decoder for next-token prediction
            logits = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=None,  # No cross-attention
                cross_attn_mask=None,
                is_causal=True,  # Decoder is always causal
            )
            return logits
        
        # Stage 2: Full training with cross-attention
        # (B, N, hidden) - N reasoning vectors from embedder representing different reasoning components
        reasoning_vectors = self.embedder(embed_input_ids, embed_attention_mask=embed_attention_mask)
        
        # Use reasoning vectors as encoder hidden states for cross-attention
        # The decoder can attend to different reasoning components
        encoder_hidden_states = reasoning_vectors  # (B, N, hidden)
        # Create cross-attention mask for all N reasoning vectors
        cross_attn_mask = torch.ones((reasoning_vectors.size(0), reasoning_vectors.size(1)), 
                                   dtype=torch.long, device=reasoning_vectors.device)  # (B, N)
        logits = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            cross_attn_mask=cross_attn_mask,
            is_causal=True,  # Decoder is always causal
        )
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,              # (B, T_prompt)
        embed_input_ids: Optional[torch.Tensor] = None,     # (B, T_plan) - None for Stage 1
        attention_mask: Optional[torch.Tensor] = None,      # (B, T_prompt)
        embed_attention_mask: Optional[torch.Tensor] = None,# (B, T_plan)
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive decoding with self-attn KV cache on the executor.
        Cross-attention reuses the same encoder_hidden_states each step.
        """
        device = input_ids.device
        B = input_ids.size(0)

        # Handle Stage 1 vs Stage 2 inference
        if embed_input_ids is None:
            # Stage 1: decoder-only inference (no cross-attention)
            encoder_hidden_states = None
            cross_attn_mask = None
        else:
            # Stage 2: full inference with cross-attention
            # 1) Compute the plan embedding once
            plan_embed = self.embedder(embed_input_ids, embed_attention_mask=embed_attention_mask)  # (B,N,H) - N reasoning vectors
            # Keep all N reasoning vectors for cross-attention
            encoder_hidden_states = plan_embed  # (B,N,H) - Keep all N reasoning vectors
            cross_attn_mask = torch.ones((B, plan_embed.size(1)), dtype=torch.long, device=device)  # (B,N)

        # 2) Prime the cache by running the full prompt once
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        logits, past_key_values = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states.expand(-1, input_ids.size(1), -1) if encoder_hidden_states is not None else None,
            cross_attn_mask=cross_attn_mask.expand(-1, input_ids.size(1)) if cross_attn_mask is not None else None,
            past_key_values=None,
            use_cache=True,
            is_causal=True,  # Decoder is always causal
        )

        generated = [input_ids]  # list of tensors to cat at the end
        cur_token = input_ids[:, -1:]  # last token of prompt (not strictly needed but keeps shape logic tidy)

        # Helper: sample one step
        def sample_from_logits(step_logits):
            if temperature != 1.0:
                step_logits = step_logits / max(1e-6, temperature)
            probs = torch.softmax(step_logits, dim=-1)

            if top_k is not None and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                mask = torch.full_like(probs, 0.0)
                mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > top_p).float()
                # keep at least 1 token
                cutoff[..., 0] = 0.0
                sorted_probs = sorted_probs * (1.0 - cutoff)
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
                # map back
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

            return torch.multinomial(probs, num_samples=1)

        # 3) Decode new tokens
        for _ in range(max_new_tokens):
            # Only feed the last generated token this step
            step_input = cur_token  # (B,1)
            step_mask = torch.ones((B, 1), dtype=torch.long, device=device)

            step_logits, past_key_values = self.decoder(
                input_ids=step_input,
                attention_mask=step_mask,
                encoder_hidden_states=encoder_hidden_states,   # (B,1,H) or None for Stage 1
                cross_attn_mask=cross_attn_mask,               # (B,1) or None for Stage 1
                past_key_values=past_key_values,
                use_cache=True,
                is_causal=True,  # Decoder is always causal
            )
            step_logits = step_logits[:, -1, :]  # (B, V)

            if do_sample:
                next_token = sample_from_logits(step_logits)
            else:
                next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

            # Append
            generated.append(next_token)
            cur_token = next_token

            # EOS handling
            if eos_token_id is not None:
                if torch.all(next_token.squeeze(-1) == eos_token_id):
                    break

        return torch.cat(generated, dim=1)  # (B, T_prompt + gen)


class PretrainedPlanDecoder(nn.Module):
    """
    Plan Decoder using pre-trained HuggingFace models.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 tokenizer,
                 pretrained_model_name: str = "qwen3/qwen3-8b",
                 device: str = "auto"):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pretrained_model_name = pretrained_model_name
        self.device = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the pre-trained model."""
        logging.info(f"Loading pre-trained plan decoder: {self.pretrained_model_name}")
        
        # Load pre-trained model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Validate tokenizer compatibility
        if not hasattr(self.tokenizer, 'vocab_size'):
            raise ValueError("Tokenizer must have vocab_size attribute")
        
        # Resize model embeddings if necessary
        if hasattr(self.model, 'resize_token_embeddings'):
            model_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            if len(self.tokenizer) != model_vocab_size:
                logging.info(f"Resizing model embeddings from {model_vocab_size} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        logging.info(f"Pre-trained plan decoder loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logging.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def generate_plan(self, 
                     prompt: str,
                     max_new_tokens: int = 256,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     do_sample: bool = True,
                     **kwargs) -> str:
        """Generate a planning sequence from a task prompt."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            plan = generated_text[len(prompt):].strip()
            
            return plan
            
        except Exception as e:
            logging.error(f"Error generating plan: {e}")
            return f"Error generating plan: {str(e)}"
    
    def forward(self, 
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """Forward pass through the pre-trained model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()


class CustomPlanDecoder(nn.Module):
    """
    Plan Decoder using custom LMHeadDecoder architecture.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 tokenizer,
                 device: str = "auto"):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the custom model using LMHeadDecoder."""
        logging.info("Loading custom plan decoder architecture")
        
        # Validate that config contains required fields
        if 'decoder_config' not in self.config:
            raise ValueError("Config must contain 'decoder_config' field for custom model")
        if 'vocab_size' not in self.config:
            raise ValueError("Config must contain 'vocab_size' field for custom model")
        
        decoder_config = self.config['decoder_config']
        vocab_size = self.config['vocab_size']
        
        # Validate tokenizer compatibility
        if not hasattr(self.tokenizer, 'vocab_size'):
            raise ValueError("Tokenizer must have vocab_size attribute")
        if len(self.tokenizer) != vocab_size:
            raise ValueError(f"Tokenizer vocab size ({len(self.tokenizer)}) does not match config vocab_size ({vocab_size})")
        
        logging.info(f"Using modular model decoder config: hidden_size={decoder_config.hidden_size}, "
                    f"num_layers={decoder_config.num_hidden_layers}, vocab_size={vocab_size}")
        
        self.model = LMHeadDecoder(decoder_config, vocab_size)
        self.model.to(self.device)
        logging.info(f"Custom plan decoder loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logging.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def generate_plan(self, 
                     prompt: str,
                     max_new_tokens: int = 256,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     do_sample: bool = True,
                     **kwargs) -> str:
        """Generate a planning sequence from a task prompt using custom model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate using custom generation method
            generated_ids = self._custom_generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample
            )
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            plan = generated_text[len(prompt):].strip()
            
            return plan
            
        except Exception as e:
            logging.error(f"Error generating plan: {e}")
            return f"Error generating plan: {str(e)}"
    
    def _custom_generate(self, inputs, max_new_tokens, temperature, top_p, top_k, do_sample):
        """Custom generation method for the LMHeadDecoder."""
        batch_size = inputs.shape[0]
        generated = inputs.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy decode
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS token
            if self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).any():
                    break
        
        return generated
    
    def forward(self, 
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """Forward pass through the custom model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()


def create_plan_decoder(config: Dict[str, Any], 
                       tokenizer,
                       use_pretrained: bool = True,
                       pretrained_model_name: str = "qwen3/qwen3-8b",
                       device: str = "auto") -> nn.Module:
    """
    Factory function to create the appropriate PlanDecoder.
    
    Args:
        config: Configuration dictionary containing model settings
        tokenizer: Tokenizer instance
        use_pretrained: Whether to use pre-trained model (True) or custom decoder (False)
        pretrained_model_name: HuggingFace model name (only used if use_pretrained=True)
        device: Device to load model on
    
    Returns:
        PretrainedPlanDecoder or CustomPlanDecoder instance
    """
    if use_pretrained:
        return PretrainedPlanDecoder(
            config=config,
            tokenizer=tokenizer,
            pretrained_model_name=pretrained_model_name,
            device=device
        )
    else:
        return CustomPlanDecoder(
            config=config,
            tokenizer=tokenizer,
            device=device
        )


# Legacy PlanDecoder class for backward compatibility
class PlanDecoder(nn.Module):
    """
    Legacy PlanDecoder class for backward compatibility.
    Use create_plan_decoder() factory function for new code.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 tokenizer,
                 use_pretrained: bool = True,
                 pretrained_model_name: str = "qwen3/qwen3-8b",
                 device: str = "auto"):
        super().__init__()
        self._decoder = create_plan_decoder(
            config=config,
            tokenizer=tokenizer,
            use_pretrained=use_pretrained,
            pretrained_model_name=pretrained_model_name,
            device=device
        )
    
    def generate_plan(self, *args, **kwargs):
        return self._decoder.generate_plan(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self._decoder.forward(*args, **kwargs)
    
    def gradient_checkpointing_enable(self):
        return self._decoder.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        return self._decoder.gradient_checkpointing_disable()


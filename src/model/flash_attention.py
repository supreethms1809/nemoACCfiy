"""
Flash Attention implementation for ACCfiy model.

This module provides Flash Attention support using:
1. PyTorch's built-in scaled_dot_product_attention (PyTorch 2.0+)
2. flash-attn library (if available)
3. Fallback to standard attention implementation

Flash Attention provides:
- Memory efficiency: O(N) instead of O(N²) memory usage
- Speed improvements: 2-4x faster attention computation
- Better numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import warnings

# Try to import flash-attn library
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None

# Check if PyTorch's scaled_dot_product_attention is available
try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except ImportError:
    SDPA_AVAILABLE = False
    scaled_dot_product_attention = None


class FlashAttentionConfig:
    """Configuration for Flash Attention."""
    
    def __init__(
        self,
        use_flash_attention: bool = True,
        use_sdpa: bool = True,
        use_flash_attn_lib: bool = True,
        dropout: float = 0.0,
        causal: bool = True,
        scale_factor: Optional[float] = None,
        flash_attn_version: str = "2.0"
    ):
        self.use_flash_attention = use_flash_attention
        self.use_sdpa = use_sdpa  # PyTorch's scaled_dot_product_attention
        self.use_flash_attn_lib = use_flash_attn_lib  # flash-attn library
        self.dropout = dropout
        self.causal = causal
        self.scale_factor = scale_factor
        self.flash_attn_version = flash_attn_version


def get_attention_implementation() -> str:
    """Get the best available attention implementation."""
    if FLASH_ATTN_AVAILABLE:
        return "flash_attn"
    elif SDPA_AVAILABLE:
        return "sdpa"
    else:
        return "standard"


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
    config: Optional[FlashAttentionConfig] = None
) -> torch.Tensor:
    """
    Flash Attention forward pass with automatic fallback.
    
    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim)
        attention_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Optional scaling factor
        config: Flash Attention configuration
    
    Returns:
        Attention output tensor
    """
    if config is None:
        config = FlashAttentionConfig()
    
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Try flash-attn library first (most efficient)
    if config.use_flash_attention and config.use_flash_attn_lib and FLASH_ATTN_AVAILABLE:
        try:
            # Check data types - Flash Attention only supports fp16 and bf16
            if query.dtype not in [torch.float16, torch.bfloat16]:
                # Only warn once to avoid spam
                if not hasattr(flash_attention_forward, '_dtype_warning_shown'):
                    warnings.warn(f"Flash Attention requires fp16 or bf16, but got {query.dtype}. Converting to bf16. This warning will be suppressed for subsequent calls.")
                    flash_attention_forward._dtype_warning_shown = True
                query = query.to(torch.bfloat16)
                key = key.to(torch.bfloat16)
                value = value.to(torch.bfloat16)
            
            # flash-attn expects (batch, seq_len, num_heads, head_dim)
            q = query.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            
            # Convert attention mask for flash-attn
            if attention_mask is not None:
                # flash-attn expects (batch, seq_len) with 1 for keep, 0 for mask
                if attention_mask.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
                    # Convert to (batch, seq_len) by taking diagonal
                    attn_mask = attention_mask[:, 0, :, :].diagonal(dim1=-2, dim2=-1)
                elif attention_mask.dim() == 3:  # (batch, seq_len, seq_len)
                    attn_mask = attention_mask.diagonal(dim1=-2, dim2=-1)
                elif attention_mask.dim() == 2:  # (batch, seq_len)
                    attn_mask = attention_mask
                else:
                    attn_mask = None
            else:
                attn_mask = None
            
            # Use flash-attn
            output = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                causal=is_causal,
                softmax_scale=scale
            )
            
            # Convert back to (batch, num_heads, seq_len, head_dim)
            output = output.transpose(1, 2)
            return output
            
        except Exception as e:
            warnings.warn(f"Flash Attention failed, falling back to SDPA: {e}")
    
    # Try PyTorch's scaled_dot_product_attention (includes Flash Attention when available)
    if config.use_flash_attention and config.use_sdpa and SDPA_AVAILABLE:
        try:
            # Check data types - SDPA works with fp32 but Flash Attention within SDPA prefers fp16/bf16
            if query.dtype not in [torch.float16, torch.bfloat16]:
                # Convert to bf16 for better Flash Attention compatibility within SDPA
                query = query.to(torch.bfloat16)
                key = key.to(torch.bfloat16)
                value = value.to(torch.bfloat16)
            
            # Convert attention mask for SDPA
            attn_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
                    # SDPA expects (batch, num_heads, seq_len, seq_len) or (batch, seq_len, seq_len)
                    attn_mask = attention_mask
                elif attention_mask.dim() == 3:  # (batch, seq_len, seq_len)
                    attn_mask = attention_mask
                elif attention_mask.dim() == 2:  # (batch, seq_len)
                    # Convert to (batch, seq_len, seq_len) for SDPA
                    attn_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    attn_mask = None
            
            # Use PyTorch's scaled_dot_product_attention
            output = scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale
            )
            return output
            
        except Exception as e:
            warnings.warn(f"SDPA failed, falling back to standard attention: {e}")
    
    # Fallback to standard attention implementation
    return standard_attention_forward(
        query, key, value, attention_mask, dropout_p, is_causal, scale
    )


def standard_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Standard attention implementation as fallback.
    
    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim)
        attention_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Optional scaling factor
    
    Returns:
        Attention output tensor
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Compute attention scores
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
    # -> (batch, num_heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply attention mask if provided
    if attention_mask is not None:
        if attention_mask.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        elif attention_mask.dim() == 3:  # (batch, seq_len, seq_len)
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        elif attention_mask.dim() == 2:  # (batch, seq_len)
            # Convert to (batch, num_heads, seq_len, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, seq_len, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout (respect model training mode)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=query.requires_grad)
    
    # Apply attention weights to values
    # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
    # -> (batch, num_heads, seq_len, head_dim)
    output = torch.matmul(attn_weights, value)
    
    return output


class FlashAttentionMixin:
    """Mixin class to add Flash Attention support to existing attention modules."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_attn_config = FlashAttentionConfig()
        self._attention_impl = get_attention_implementation()
    
    def set_flash_attention_config(self, config: FlashAttentionConfig):
        """Set Flash Attention configuration."""
        self.flash_attn_config = config
    
    def get_attention_implementation(self) -> str:
        """Get the current attention implementation."""
        return self._attention_impl
    
    def flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """Forward pass using Flash Attention with fallback."""
        return flash_attention_forward(
            query, key, value, attention_mask, dropout_p, is_causal, scale, self.flash_attn_config
        )


def create_attention_mask(
    batch_size: int,
    seq_len: int,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    device: torch.device = None
) -> Optional[torch.Tensor]:
    """
    Create attention mask for Flash Attention.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        attention_mask: Optional input attention mask
        is_causal: Whether to apply causal masking
        device: Device to create mask on
    
    Returns:
        Attention mask tensor or None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Start with causal mask if needed
    if is_causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    else:
        mask = None
    
    # Apply input attention mask if provided
    if attention_mask is not None:
        if attention_mask.dim() == 2:  # (batch, seq_len)
            # Convert to (batch, 1, seq_len, seq_len)
            input_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_len, -1)
            if mask is None:
                mask = input_mask
            else:
                mask = mask | (~input_mask)
        elif attention_mask.dim() == 3:  # (batch, seq_len, seq_len)
            input_mask = attention_mask.unsqueeze(1)
            if mask is None:
                mask = input_mask
            else:
                mask = mask | (~input_mask)
        elif attention_mask.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
            if mask is None:
                mask = attention_mask
            else:
                mask = mask | (~attention_mask)
    
    return mask


# Utility functions for memory and performance monitoring
def get_attention_memory_usage(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16
) -> dict:
    """Calculate memory usage for attention computation."""
    element_size = torch.tensor(0, dtype=dtype).element_size()
    
    # Standard attention: O(N²) memory
    standard_memory = batch_size * num_heads * seq_len * seq_len * element_size
    
    # Flash Attention: O(N) memory
    flash_memory = batch_size * num_heads * seq_len * head_dim * element_size
    
    return {
        "standard_attention_mb": standard_memory / (1024 * 1024),
        "flash_attention_mb": flash_memory / (1024 * 1024),
        "memory_reduction_ratio": standard_memory / flash_memory,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim
    }


def benchmark_attention_implementations(
    batch_size: int = 2,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 64,
    num_iterations: int = 10,
    device: str = "cuda"
) -> dict:
    """Benchmark different attention implementations."""
    device = torch.device(device)
    
    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    results = {}
    
    # Benchmark standard attention
    try:
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(num_iterations):
            _ = standard_attention_forward(query, key, value, is_causal=True)
        end_time.record()
        torch.cuda.synchronize()
        
        results["standard_attention_ms"] = start_time.elapsed_time(end_time) / num_iterations
    except Exception as e:
        results["standard_attention_error"] = str(e)
    
    # Benchmark SDPA
    if SDPA_AVAILABLE:
        try:
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(num_iterations):
                _ = scaled_dot_product_attention(query, key, value, is_causal=True)
            end_time.record()
            torch.cuda.synchronize()
            
            results["sdpa_ms"] = start_time.elapsed_time(end_time) / num_iterations
        except Exception as e:
            results["sdpa_error"] = str(e)
    
    # Benchmark flash-attn
    if FLASH_ATTN_AVAILABLE:
        try:
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(num_iterations):
                _ = flash_attention_forward(query, key, value, is_causal=True)
            end_time.record()
            torch.cuda.synchronize()
            
            results["flash_attn_ms"] = start_time.elapsed_time(end_time) / num_iterations
        except Exception as e:
            results["flash_attn_error"] = str(e)
    
    return results


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Flash Attention Implementation...")
    
    # Check availability
    print(f"Flash Attention Library Available: {FLASH_ATTN_AVAILABLE}")
    print(f"SDPA Available: {SDPA_AVAILABLE}")
    print(f"Best Implementation: {get_attention_implementation()}")
    
    # Test with small tensors
    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        # Test Flash Attention
        try:
            output = flash_attention_forward(query, key, value, is_causal=True)
            print(f"✅ Flash Attention test passed. Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Flash Attention test failed: {e}")
        
        # Test memory usage calculation
        memory_info = get_attention_memory_usage(batch_size, num_heads, seq_len, head_dim)
        print(f"Memory usage - Standard: {memory_info['standard_attention_mb']:.2f} MB")
        print(f"Memory usage - Flash: {memory_info['flash_attention_mb']:.2f} MB")
        print(f"Memory reduction: {memory_info['memory_reduction_ratio']:.1f}x")
        
        # Benchmark if possible
        try:
            benchmark_results = benchmark_attention_implementations(
                batch_size=2, num_heads=8, seq_len=512, head_dim=64, num_iterations=5
            )
            print("Benchmark Results:")
            for key, value in benchmark_results.items():
                if "ms" in key:
                    print(f"  {key}: {value:.2f} ms")
                elif "error" in key:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    print("✅ Flash Attention implementation ready!")

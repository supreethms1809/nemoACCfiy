#!/usr/bin/env python3
"""
Tokenizer Manager for NeMo ModularModel

This module handles tokenizer downloading, special token addition, and local caching
to avoid re-downloading tokenizers on every run.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TokenizerManager:
    """Manages tokenizer downloading, customization, and caching."""
    
    def __init__(self, cache_dir: str = "tokenizers", base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"):
        """
        Initialize the tokenizer manager.
        
        Args:
            cache_dir: Directory to cache tokenizers
            base_model: Base HuggingFace model to download tokenizer from
        """
        self.cache_dir = Path(cache_dir)
        self.base_model = base_model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default special tokens to add (can be customized)
        self.default_special_tokens = [
            "<plan>",      # For planning/reasoning sections
            "</plan>",     # End of planning section
            "<reasoning>", # For reasoning steps
            "</reasoning>", # End of reasoning
            "<code>",      # For code blocks
            "</code>",     # End of code blocks
            "<output>",    # For expected outputs
            "</output>",   # End of expected outputs
        ]
        
        # Metadata file to track cached tokenizers
        self.metadata_file = self.cache_dir / "tokenizer_metadata.json"
    
    def get_cached_tokenizer_path(self, custom_tokens: Optional[List[str]] = None) -> Path:
        """
        Get the path for a cached tokenizer with specific custom tokens.
        
        Args:
            custom_tokens: List of custom tokens to add (None for default)
            
        Returns:
            Path to the cached tokenizer directory
        """
        if custom_tokens is None:
            custom_tokens = self.default_special_tokens
        
        # Create a deterministic hash of the custom tokens for the directory name
        import hashlib
        tokens_str = "|".join(sorted(custom_tokens))
        tokens_hash = hashlib.md5(tokens_str.encode()).hexdigest()[:8]
        tokenizer_name = f"qwen3-coder-custom-{tokens_hash}"
        return self.cache_dir / tokenizer_name
    
    def is_tokenizer_cached(self, custom_tokens: Optional[List[str]] = None) -> bool:
        """
        Check if a tokenizer with specific custom tokens is already cached.
        
        Args:
            custom_tokens: List of custom tokens to check for
            
        Returns:
            True if tokenizer is cached, False otherwise
        """
        cached_path = self.get_cached_tokenizer_path(custom_tokens)
        return cached_path.exists() and (cached_path / "tokenizer_config.json").exists()
    
    def load_cached_tokenizer(self, custom_tokens: Optional[List[str]] = None) -> AutoTokenizer:
        """
        Load a cached tokenizer.
        
        Args:
            custom_tokens: List of custom tokens (must match cached version)
            
        Returns:
            Loaded tokenizer
            
        Raises:
            FileNotFoundError: If tokenizer is not cached
        """
        cached_path = self.get_cached_tokenizer_path(custom_tokens)
        
        if not self.is_tokenizer_cached(custom_tokens):
            raise FileNotFoundError(f"Tokenizer not cached at {cached_path}")
        
        logger.info(f"Loading cached tokenizer from: {cached_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(cached_path))
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"✅ Cached tokenizer loaded with vocab size: {len(tokenizer)}")
        return tokenizer
    
    def download_and_customize_tokenizer(self, custom_tokens: Optional[List[str]] = None) -> AutoTokenizer:
        """
        Download tokenizer from HuggingFace Hub and add custom tokens.
        
        Args:
            custom_tokens: List of custom tokens to add (None for default)
            
        Returns:
            Customized tokenizer
        """
        if custom_tokens is None:
            custom_tokens = self.default_special_tokens
        
        logger.info(f"Downloading tokenizer from HuggingFace Hub: {self.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Add custom tokens
        if custom_tokens:
            logger.info(f"Adding {len(custom_tokens)} custom tokens: {custom_tokens}")
            tokenizer.add_tokens(custom_tokens)
            logger.info(f"✅ Added custom tokens. New vocab size: {len(tokenizer)}")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        return tokenizer
    
    def save_tokenizer(self, tokenizer: AutoTokenizer, custom_tokens: Optional[List[str]] = None) -> Path:
        """
        Save a customized tokenizer to cache.
        
        Args:
            tokenizer: Tokenizer to save
            custom_tokens: List of custom tokens that were added
            
        Returns:
            Path where tokenizer was saved
        """
        if custom_tokens is None:
            custom_tokens = self.default_special_tokens
        
        cached_path = self.get_cached_tokenizer_path(custom_tokens)
        
        logger.info(f"Saving tokenizer to cache: {cached_path}")
        tokenizer.save_pretrained(str(cached_path))
        
        # Save metadata
        self._save_metadata(cached_path, custom_tokens)
        
        logger.info(f"✅ Tokenizer saved to cache with vocab size: {len(tokenizer)}")
        return cached_path
    
    def _save_metadata(self, tokenizer_path: Path, custom_tokens: List[str]):
        """Save metadata about the cached tokenizer."""
        metadata = {
            "base_model": self.base_model,
            "custom_tokens": custom_tokens,
            "vocab_size": len(AutoTokenizer.from_pretrained(str(tokenizer_path))),
            "cached_at": str(tokenizer_path),
        }
        
        # Load existing metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        # Add new metadata
        all_metadata[str(tokenizer_path)] = metadata
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
    
    def get_tokenizer(self, custom_tokens: Optional[List[str]] = None, force_download: bool = False) -> AutoTokenizer:
        """
        Get a tokenizer, using cache if available or downloading and customizing if needed.
        
        Args:
            custom_tokens: List of custom tokens to add (None for default)
            force_download: Force re-download even if cached
            
        Returns:
            Tokenizer with custom tokens
        """
        if custom_tokens is None:
            custom_tokens = self.default_special_tokens
        
        # Check if we can use cached version
        if not force_download and self.is_tokenizer_cached(custom_tokens):
            try:
                return self.load_cached_tokenizer(custom_tokens)
            except Exception as e:
                logger.warning(f"Failed to load cached tokenizer: {e}. Re-downloading...")
        
        # Download and customize
        tokenizer = self.download_and_customize_tokenizer(custom_tokens)
        
        # Save to cache
        try:
            self.save_tokenizer(tokenizer, custom_tokens)
        except Exception as e:
            logger.warning(f"Failed to save tokenizer to cache: {e}")
        
        return tokenizer
    
    def list_cached_tokenizers(self) -> Dict[str, Dict[str, Any]]:
        """List all cached tokenizers and their metadata."""
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def clear_cache(self, keep_latest: bool = True):
        """
        Clear cached tokenizers.
        
        Args:
            keep_latest: If True, keep the most recently used tokenizer
        """
        if not self.metadata_file.exists():
            logger.info("No cached tokenizers to clear")
            return
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if keep_latest and len(metadata) > 1:
            # Keep the most recent one (simple heuristic: keep the first one)
            tokenizers_to_remove = list(metadata.keys())[1:]
        else:
            tokenizers_to_remove = list(metadata.keys())
        
        for tokenizer_path in tokenizers_to_remove:
            try:
                import shutil
                shutil.rmtree(tokenizer_path)
                logger.info(f"Removed cached tokenizer: {tokenizer_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {tokenizer_path}: {e}")
        
        # Update metadata file
        if keep_latest and len(metadata) > 1:
            remaining = {k: v for k, v in metadata.items() if k not in tokenizers_to_remove}
            with open(self.metadata_file, 'w') as f:
                json.dump(remaining, f, indent=2)
        else:
            self.metadata_file.unlink()


def get_tokenizer_with_caching(
    tokenizer_path: str = "tokenizers/qwen3-coder-30b-a3b-instruct-custom",
    custom_tokens: Optional[List[str]] = None,
    force_download: bool = False,
    cache_dir: str = "tokenizers"
) -> AutoTokenizer:
    """
    Convenience function to get a tokenizer with caching support.
    
    Args:
        tokenizer_path: Expected local path (used for compatibility)
        custom_tokens: List of custom tokens to add
        force_download: Force re-download even if cached
        cache_dir: Directory for tokenizer cache
        
    Returns:
        Tokenizer with custom tokens
    """
    manager = TokenizerManager(cache_dir=cache_dir)
    
    # Check if the expected local path exists
    if os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from local path: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    # Use cached/downloaded tokenizer
    return manager.get_tokenizer(custom_tokens=custom_tokens, force_download=force_download)

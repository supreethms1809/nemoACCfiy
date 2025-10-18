"""
NeMo Integration Package for ModularModel

This package provides a complete NeMo integration for the ModularModel,
including proper NeMo module registration, configuration management, and training integration.
"""

from .nemo_wrapper import (
    create_modular_model_nemo,
    ModularModelConfig,
    ModularModelNeMo,
    ModularModelNeMoWrapper,
)

__version__ = "1.0.0"
__author__ = "ACCfiy Team"

__all__ = [
    "create_modular_model_nemo",
    "ModularModelConfig", 
    "ModularModelNeMo",
    "ModularModelNeMoWrapper",
]

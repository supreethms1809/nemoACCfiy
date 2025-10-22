"""
NeMo Integration Package for ModularModel

This package provides a complete NeMo integration for the ModularModel,
including proper NeMo module registration, configuration management, and training integration.
"""

# Only import NeMo components if NeMo is available
try:
    from .nemo_wrapper import (
        create_modular_model_nemo,
        ModularModelConfig,
        ModularModelNeMo,
        ModularModelNeMoWrapper,
    )
    NEMO_COMPONENTS_AVAILABLE = True
except ImportError:
    # NeMo not available, create placeholder classes
    NEMO_COMPONENTS_AVAILABLE = False
    
    class ModularModelConfig:
        pass
    
    class ModularModelNeMo:
        pass
    
    class ModularModelNeMoWrapper:
        pass
    
    def create_modular_model_nemo(*args, **kwargs):
        raise RuntimeError("NeMo not available. Please install NeMo to use this functionality.")

__version__ = "1.0.0"
__author__ = "ACCfiy Team"

__all__ = [
    "create_modular_model_nemo",
    "ModularModelConfig", 
    "ModularModelNeMo",
    "ModularModelNeMoWrapper",
]

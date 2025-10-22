#!/usr/bin/env python3
"""
Unified Training Entry Point for NeMo ModularModel

This script provides a single entry point for all training modes:
- Basic training (simple datasets)
- Production training (configuration-driven)

Usage:
    python train.py --mode basic --stage stage1
    python train.py --mode production --model_config model_config_1.8B --stage stage1
"""

import sys
import os
from pathlib import Path

# Add project root to system path for consistent imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nemo.ModularModelstage1_NTPtraining import main

if __name__ == "__main__":
    main()

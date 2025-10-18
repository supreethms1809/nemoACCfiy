#!/usr/bin/env python3
"""
Unified Training Entry Point for NeMo ModularModel

This script provides a single entry point for all training modes:
- Basic training (simple datasets)
- Production training (configuration-driven) 
- Foundation training (NeMo native datasets)

Usage:
    python train.py --mode basic --stage stage1
    python train.py --mode production --model_config model_config_1.7B --stage stage1
    python train.py --mode foundation --data_path ./data --stage stage1
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.nemo.ModularModelstage1_NTPtraining import main

if __name__ == "__main__":
    main()

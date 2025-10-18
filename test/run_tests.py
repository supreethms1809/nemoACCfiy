#!/usr/bin/env python3
"""
Simple test runner script for NeMo training tests.
This script runs the tests from the project root directory.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path (since we're now in the test directory)
test_dir = Path(__file__).parent
sys.path.append(str(test_dir))

# Import and run the test runner
from run_nemo_tests import main

if __name__ == "__main__":
    main()

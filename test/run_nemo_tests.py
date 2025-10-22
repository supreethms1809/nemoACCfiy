#!/usr/bin/env python3
"""
Test runner for NeMo training tests.
This script provides an easy way to run different NeMo training tests.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the current directory to the path (we're already in the test directory)
test_dir = Path(__file__).parent
sys.path.append(str(test_dir))

def run_comprehensive_test():
    """Run the comprehensive NeMo Stage 1 training test."""
    print("Running comprehensive NeMo Stage 1 training test...")
    try:
        from test_comprehensive import test_nemo_stage1_training_comprehensive
        return test_nemo_stage1_training_comprehensive()
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_basic_test():
    """Run a basic NeMo training test."""
    print("Running basic NeMo training test...")
    try:
        from test_comprehensive import test_nemo_stage1_training
        return test_nemo_stage1_training()
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fsdp_test():
    """Run FSDP integration test."""
    print("Running FSDP integration test...")
    try:
        from test_comprehensive import test_nemo_fsdp_integration
        return test_nemo_fsdp_integration()
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_production_test():
    """Run production training test."""
    print("Running production training test...")
    try:
        from test_comprehensive import test_nemo_production_training
        return test_nemo_production_training()
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_padding_test():
    """Run padding and target correctness test."""
    print("Running padding and target correctness test...")
    try:
        from test_comprehensive import test_nemo_padding_edge_cases
        from transformers import AutoTokenizer
        
        # Create a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create a dummy training module (we only need it for the test signature)
        class DummyTrainingModule:
            pass
        
        training_module = DummyTrainingModule()
        
        return test_nemo_padding_edge_cases(tokenizer, training_module)
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="NeMo Training Test Runner")
    parser.add_argument("--test", type=str, default="comprehensive",
                       choices=["comprehensive", "basic", "fsdp", "production", "padding", "all"],
                       help="Type of test to run")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("="*80)
    print("NEMO TRAINING TEST RUNNER")
    print("="*80)
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    success = True
    
    if args.test == "comprehensive":
        success = run_comprehensive_test()
    elif args.test == "basic":
        success = run_basic_test()
    elif args.test == "fsdp":
        success = run_fsdp_test()
    elif args.test == "production":
        success = run_production_test()
    elif args.test == "padding":
        success = run_padding_test()
    elif args.test == "all":
        print("Running all tests...")
        tests = [
            ("Basic Test", run_basic_test),
            ("FSDP Test", run_fsdp_test),
            ("Production Test", run_production_test),
            ("Padding Test", run_padding_test),
            ("Comprehensive Test", run_comprehensive_test)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"Running {test_name}")
            print(f"{'='*60}")
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name}: {'✅ PASSED' if result else '❌ FAILED'}")
        
        print(f"\n{'='*60}")
        print("ALL TESTS SUMMARY")
        print(f"{'='*60}")
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        success = all(result for _, result in results)
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ NeMo training is working correctly")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There are issues with NeMo training implementation")
        sys.exit(1)
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

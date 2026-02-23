#!/usr/bin/env python
"""
Run all test suites for the traffic wave reconstruction project.

Usage:
    python src/tests/run_all_tests.py
    python src/tests/run_all_tests.py --verbose
"""
from __future__ import annotations

import argparse
import sys
import traceback


def run_test_module(module_name: str, verbose: bool = False) -> bool:
    """Run a test module and return success status."""
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {module_name}")
            print(f"{'='*60}")

        module = __import__(module_name, fromlist=["run_all_tests"])
        module.run_all_tests()
        return True
    except AssertionError as e:
        print(f"\nFAILED: {module_name}")
        print(f"  {e}")
        if verbose:
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"\nERROR: {module_name}")
        print(f"  {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    test_modules = [
        "src.tests.test_phase1",
        "src.tests.test_reparam",
        "src.tests.test_graph_prior",
        "src.tests.test_metrics",
        "src.tests.test_integration",
    ]

    print("=" * 70)
    print("Running All Test Suites")
    print("=" * 70)

    results = {}
    for module in test_modules:
        results[module] = run_test_module(module, args.verbose)

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for module, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  {module}: {status}")

    print("-" * 70)
    print(f"Total: {passed}/{total} passed")

    if passed == total:
        print("\nAll tests PASSED!")
        return 0
    else:
        print(f"\n{total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

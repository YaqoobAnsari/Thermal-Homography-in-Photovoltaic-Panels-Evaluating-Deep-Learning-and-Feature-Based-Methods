"""
run_all_traditional_methods.py

Master script to run all traditional homography estimation methods sequentially.
Runs: SIFT, ORB, AKAZE, BRISK, KAZE
Automatically generates comparison results after all methods complete.

Author: Homography Benchmarking Project
Date: 2025
"""

import os
import sys
import time

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import traditional methods from Utils
from Utils.traditional_sift import SIFTHomographyEstimator
from Utils.traditional_orb import ORBHomographyEstimator
from Utils.traditional_akaze import AKAZEHomographyEstimator
from Utils.traditional_brisk import BRISKHomographyEstimator
from Utils.traditional_kaze import KAZEHomographyEstimator

# Import comparison generation utilities
from Utils.generate_comparison_results import (
    generate_cross_method_comparison,
    generate_performance_matrix_csv,
    generate_per_datatype_comparison,
    generate_category_analysis,
    print_category_insights
)

# Methods to run (in order)
METHODS = {
    'SIFT': SIFTHomographyEstimator,
    'ORB': ORBHomographyEstimator,
    'AKAZE': AKAZEHomographyEstimator,
    'BRISK': BRISKHomographyEstimator,
    'KAZE': KAZEHomographyEstimator,
}

# Paths
HOMOGRAPHY_PAIRS_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Homography Pairs')
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results')


def run_method(method_name, estimator_class):
    """
    Run a single traditional method.

    Args:
        method_name: Name of the method (e.g., 'SIFT')
        estimator_class: Class of the estimator

    Returns:
        Tuple of (success, elapsed_time)
    """
    print(f"\n{'='*80}")
    print(f"RUNNING: {method_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Create estimator instance
        estimator = estimator_class()

        # Set output directory
        output_dir = os.path.join(RESULTS_BASE_DIR, method_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get all data type directories
        data_types = [d for d in os.listdir(HOMOGRAPHY_PAIRS_DIR)
                     if os.path.isdir(os.path.join(HOMOGRAPHY_PAIRS_DIR, d))]

        if len(data_types) == 0:
            print(f"\nERROR: No data types found in {HOMOGRAPHY_PAIRS_DIR}")
            return False, 0

        print(f"\nFound {len(data_types)} data types to process")

        # Process each data type
        for idx, data_type in enumerate(sorted(data_types), 1):
            print(f"\n[{idx}/{len(data_types)}] Processing: {data_type}")
            data_type_dir = os.path.join(HOMOGRAPHY_PAIRS_DIR, data_type)
            estimator.process_dataset(data_type_dir, output_dir)

        # Save summary
        print(f"\nSaving summary for {method_name}...")
        summary = estimator.save_summary(output_dir)

        elapsed_time = time.time() - start_time

        print(f"\n  SUCCESS: {method_name} completed successfully")
        print(f"  Total pairs: {summary['total_pairs']}")
        print(f"  Successful: {summary['successful_pairs']} ({summary['success_rate']*100:.2f}%)")
        print(f"  Elapsed time: {elapsed_time:.2f} seconds")
        return True, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n  FAILED: {method_name} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed_time


def generate_comparisons():
    """
    Generate cross-method comparison results.

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("GENERATING COMPARISON RESULTS")
    print("="*80)

    try:
        # Check if results directory exists
        if not os.path.exists(RESULTS_BASE_DIR):
            print(f"\nERROR: Results directory not found: {RESULTS_BASE_DIR}")
            return False

        # Generate all comparisons
        cross_method = generate_cross_method_comparison()
        performance_matrix = generate_performance_matrix_csv()
        per_datatype = generate_per_datatype_comparison()
        category_analysis = generate_category_analysis()

        print("\n" + "="*80)
        print("COMPARISON RESULTS GENERATION COMPLETE")
        print("="*80)

        print("\nGenerated files:")
        print(f"  1. {os.path.join(RESULTS_BASE_DIR, 'cross_method_comparison.json')}")
        print(f"  2. {os.path.join(RESULTS_BASE_DIR, 'performance_matrix.csv')}")
        print(f"  3. {os.path.join(RESULTS_BASE_DIR, 'per_datatype_comparison.json')}")
        print(f"  4. {os.path.join(RESULTS_BASE_DIR, 'category_analysis.json')}")

        # Print quick summary
        if cross_method and 'analysis' in cross_method:
            print("\nQuick Analysis:")
            analysis = cross_method['analysis']
            print(f"  Most Accurate (MACE): {analysis.get('most_accurate_mace', 'N/A')}")
            print(f"  Most Reliable: {analysis.get('most_reliable', 'N/A')}")
            print(f"  Fastest: {analysis.get('fastest', 'N/A')}")
            print(f"  Best Precision: {analysis.get('best_precision', 'N/A')}")
            print(f"  Best F1 Score: {analysis.get('best_f1_score', 'N/A')}")
            print(f"  Best Overall: {analysis.get('best_overall', 'N/A')}")

        # Print category insights
        if category_analysis:
            print_category_insights(category_analysis)

        print("\n" + "="*80)
        return True

    except Exception as e:
        print(f"\nERROR: Failed to generate comparison results: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Run all traditional methods sequentially and generate comparisons.
    """
    print("="*80)
    print("RUNNING ALL TRADITIONAL HOMOGRAPHY ESTIMATION METHODS")
    print("="*80)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Homography pairs directory: {HOMOGRAPHY_PAIRS_DIR}")
    print(f"Results directory: {RESULTS_BASE_DIR}")
    print(f"\nMethods to run: {len(METHODS)}")
    for idx, method in enumerate(METHODS.keys(), 1):
        print(f"  {idx}. {method}")

    print("\n" + "="*80)

    # Check if homography pairs directory exists
    if not os.path.exists(HOMOGRAPHY_PAIRS_DIR):
        print(f"\nERROR: Homography pairs directory not found: {HOMOGRAPHY_PAIRS_DIR}")
        print("Please run generate_homography_pairs.py first.")
        return False

    # Create results base directory
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    # Run each method
    results = {}
    total_start_time = time.time()

    for idx, (method_name, estimator_class) in enumerate(METHODS.items(), 1):
        print(f"\n[{idx}/{len(METHODS)}] Starting {method_name}...")
        success, elapsed = run_method(method_name, estimator_class)
        results[method_name] = {'success': success, 'time': elapsed}

    total_elapsed = time.time() - total_start_time

    # Summary of method execution
    print("\n" + "="*80)
    print("METHOD EXECUTION SUMMARY")
    print("="*80)

    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful

    print(f"\nTotal methods: {len(METHODS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print(f"\nDetailed Results:")
    for method, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"  {method:15s}: {status:12s} ({result['time']:.2f}s)")

    print(f"\nTotal execution time: {total_elapsed/60:.2f} minutes ({total_elapsed:.2f} seconds)")

    # Generate comparison results if at least one method succeeded
    if successful > 0:
        comparison_success = generate_comparisons()
    else:
        comparison_success = False
        print("\nSkipping comparison generation - no methods succeeded.")

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nAll traditional methods have been executed.")
    print(f"Results saved to: {RESULTS_BASE_DIR}")

    if comparison_success:
        print(f"\nComparison results successfully generated:")
        print(f"  - Cross-method comparison JSON")
        print(f"  - Performance matrix CSV")
        print(f"  - Per-datatype comparison JSON")

    print("="*80)

    return successful == len(METHODS) and comparison_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

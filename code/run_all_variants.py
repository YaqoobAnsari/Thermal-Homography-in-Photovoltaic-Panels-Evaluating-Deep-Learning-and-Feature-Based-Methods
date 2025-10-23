"""
run_all_variants.py

Comprehensive benchmarking script that runs all traditional methods
with all RANSAC variants for complete performance analysis.

Runs 15 configurations:
- 5 Methods: SIFT, ORB, AKAZE, BRISK, KAZE
- 3 RANSAC Variants: RANSAC, MLESAC, PROSAC
= 15 total combinations

Author: Homography Benchmarking Project
Date: 2025
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from Utils.traditional_sift import SIFTHomographyEstimator
from Utils.traditional_orb import ORBHomographyEstimator
from Utils.traditional_akaze import AKAZEHomographyEstimator
from Utils.traditional_brisk import BRISKHomographyEstimator
from Utils.traditional_kaze import KAZEHomographyEstimator
from Utils.generate_comparison_results import (
    generate_cross_method_comparison,
    generate_performance_matrix_csv,
    generate_per_datatype_comparison,
    generate_category_analysis,
    generate_ransac_variant_comparison,
    print_category_insights,
    print_ransac_insights
)

# Configuration
HOMOGRAPHY_PAIRS_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Homography Pairs')
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results')

# Method configurations
METHODS = {
    'SIFT': SIFTHomographyEstimator,
    'ORB': ORBHomographyEstimator,
    'AKAZE': AKAZEHomographyEstimator,
    'BRISK': BRISKHomographyEstimator,
    'KAZE': KAZEHomographyEstimator
}

# RANSAC variant configurations
RANSAC_VARIANTS = ['RANSAC', 'MLESAC', 'PROSAC']


def run_method_with_variant(method_name, estimator_class, ransac_variant):
    """
    Run a single method with a specific RANSAC variant.

    Args:
        method_name: Name of the feature detection method
        estimator_class: Estimator class to instantiate
        ransac_variant: RANSAC variant to use ('RANSAC', 'MLESAC', 'PROSAC')

    Returns:
        Tuple of (success, elapsed_time)
    """
    # Create unique identifier for this configuration
    config_name = f"{method_name}_{ransac_variant}"

    print(f"\n{'='*80}")
    print(f"RUNNING: {config_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Create estimator with specified RANSAC variant
        estimator = estimator_class(
            ransac_method=ransac_variant,
            use_advanced_matching=True,
            distance_ratio=0.75
        )

        # Output directory for this configuration
        output_dir = os.path.join(RESULTS_BASE_DIR, config_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get all data types
        if not os.path.exists(HOMOGRAPHY_PAIRS_DIR):
            print(f"ERROR: Homography pairs directory not found: {HOMOGRAPHY_PAIRS_DIR}")
            return False, 0

        data_types = [d for d in os.listdir(HOMOGRAPHY_PAIRS_DIR)
                     if os.path.isdir(os.path.join(HOMOGRAPHY_PAIRS_DIR, d))]

        if not data_types:
            print(f"ERROR: No data types found in {HOMOGRAPHY_PAIRS_DIR}")
            return False, 0

        print(f"\nProcessing {len(data_types)} data types with {ransac_variant}...")

        # Process each data type
        for datatype in sorted(data_types):
            data_type_dir = os.path.join(HOMOGRAPHY_PAIRS_DIR, datatype)
            estimator.process_dataset(data_type_dir, output_dir)

        # Save summary
        print(f"\nGenerating summary for {config_name}...")
        summary = estimator.save_summary(output_dir)

        elapsed_time = time.time() - start_time

        print(f"\n{config_name} completed successfully!")
        print(f"  Total pairs: {summary.get('total_pairs', 0)}")
        print(f"  Successful: {summary.get('successful_pairs', 0)}")
        print(f"  Success rate: {summary.get('success_rate', 0):.2%}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")

        return True, elapsed_time

    except Exception as e:
        print(f"\nERROR in {config_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, time.time() - start_time


def main():
    """
    Main function to run all method-variant combinations.
    """
    print("="*80)
    print("COMPREHENSIVE HOMOGRAPHY BENCHMARKING")
    print("All Methods × All RANSAC Variants")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Methods: {len(METHODS)} ({', '.join(METHODS.keys())})")
    print(f"  RANSAC Variants: {len(RANSAC_VARIANTS)} ({', '.join(RANSAC_VARIANTS)})")
    print(f"  Total Combinations: {len(METHODS) * len(RANSAC_VARIANTS)}")
    print(f"\nResults will be saved to: {RESULTS_BASE_DIR}")

    # Track results
    results_summary = []
    total_start_time = time.time()

    # Run all combinations
    combination_num = 0
    for method_name, estimator_class in METHODS.items():
        for ransac_variant in RANSAC_VARIANTS:
            combination_num += 1
            config_name = f"{method_name}_{ransac_variant}"

            print(f"\n\n{'#'*80}")
            print(f"COMBINATION {combination_num}/{len(METHODS) * len(RANSAC_VARIANTS)}: {config_name}")
            print(f"{'#'*80}")

            success, elapsed = run_method_with_variant(method_name, estimator_class, ransac_variant)

            results_summary.append({
                'config': config_name,
                'method': method_name,
                'ransac_variant': ransac_variant,
                'success': success,
                'time': elapsed
            })

    # Generate comprehensive comparisons
    print("\n\n" + "="*80)
    print("GENERATING COMPREHENSIVE COMPARISONS")
    print("="*80)

    successful = sum(1 for r in results_summary if r['success'])

    if successful > 0:
        print("\nGenerating cross-method comparison...")
        cross_method = generate_cross_method_comparison()

        print("Generating performance matrix (CSV)...")
        performance_matrix = generate_performance_matrix_csv()

        print("Generating per-datatype comparison...")
        per_datatype = generate_per_datatype_comparison()

        print("Generating category analysis...")
        category_analysis = generate_category_analysis()

        print("Generating RANSAC variant comparison...")
        ransac_comparison = generate_ransac_variant_comparison()

        print("\n" + "="*80)
        print("INSIGHTS")
        print("="*80)

        print("\n--- COLORMAP CATEGORY INSIGHTS ---")
        print_category_insights(category_analysis)

        print("\n--- RANSAC VARIANT INSIGHTS ---")
        print_ransac_insights(ransac_comparison)

    # Print final summary
    total_time = time.time() - total_start_time

    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"\nResults:")
    print(f"  Total combinations: {len(results_summary)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results_summary) - successful}")

    print("\n--- Per-Configuration Results ---")
    for result in results_summary:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"  {result['config']:25s} - {status:7s} - {result['time']:6.2f}s")

    if successful == len(results_summary):
        print("\n" + "="*80)
        print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
        print("="*80)
    else:
        print(f"\nWARNING: {len(results_summary) - successful} configuration(s) failed")

    print(f"\nResults saved to: {RESULTS_BASE_DIR}")
    print("\nGenerated files:")
    print("  - cross_method_comparison.json (method + variant rankings)")
    print("  - performance_matrix.csv (20 metrics × 15 configurations)")
    print("  - per_datatype_comparison.json (29 datatypes analyzed)")
    print("  - category_analysis.json (8 colormap categories)")
    print("  - ransac_variant_comparison.json (RANSAC variant analysis)")


if __name__ == "__main__":
    main()

"""
regenerate_summaries.py

Regenerates summary files for all traditional methods using enhanced evaluation metrics.
Reads existing detailed results and creates new summaries with comprehensive metrics.

Author: Homography Benchmarking Project
Date: October 2025
"""

import os
import sys

# Add code directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from traditional_sift import SIFTHomographyEstimator
from traditional_orb import ORBHomographyEstimator
from traditional_akaze import AKAZEHomographyEstimator
from traditional_brisk import BRISKHomographyEstimator
from traditional_kaze import KAZEHomographyEstimator

# Get project paths
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results')

# Methods to regenerate
METHODS = {
    'SIFT': (SIFTHomographyEstimator, os.path.join(RESULTS_BASE_DIR, 'SIFT')),
    'ORB': (ORBHomographyEstimator, os.path.join(RESULTS_BASE_DIR, 'ORB')),
    'AKAZE': (AKAZEHomographyEstimator, os.path.join(RESULTS_BASE_DIR, 'AKAZE')),
    'BRISK': (BRISKHomographyEstimator, os.path.join(RESULTS_BASE_DIR, 'BRISK')),
    'KAZE': (KAZEHomographyEstimator, os.path.join(RESULTS_BASE_DIR, 'KAZE'))
}


def regenerate_summary(method_name, estimator_class, results_dir):
    """
    Regenerate summary files for a single method.

    Args:
        method_name: Name of the method (e.g., 'SIFT')
        estimator_class: Class for the estimator
        results_dir: Directory containing detailed results
    """
    print(f"\n{'='*80}")
    print(f"Regenerating summary for: {method_name}")
    print(f"{'='*80}")

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"  ERROR: Results directory not found: {results_dir}")
        return False

    detailed_results_dir = os.path.join(results_dir, 'detailed_results')
    if not os.path.exists(detailed_results_dir):
        print(f"  ERROR: Detailed results directory not found: {detailed_results_dir}")
        return False

    # Create estimator instance
    estimator = estimator_class()

    # Remove old summary files
    summary_file = os.path.join(results_dir, f'{method_name}_summary.json')
    per_datatype_file = os.path.join(results_dir, f'{method_name}_per_datatype_summary.json')

    if os.path.exists(summary_file):
        os.remove(summary_file)
        print(f"  Removed old summary: {summary_file}")

    if os.path.exists(per_datatype_file):
        os.remove(per_datatype_file)
        print(f"  Removed old per-datatype summary: {per_datatype_file}")

    # Regenerate summaries using enhanced metrics
    print(f"  Regenerating summaries with enhanced evaluation metrics...")
    summary = estimator.save_summary(results_dir)

    print(f"  [OK] Summary regenerated successfully")
    print(f"  Total pairs: {summary.get('total_pairs', 'N/A')}")
    print(f"  Success rate: {summary.get('success_rate', 0)*100:.2f}%")

    if 'mace_statistics' in summary:
        print(f"  Mean MACE: {summary['mace_statistics'].get('mean', 'N/A'):.2f} pixels")

    return True


def main():
    """
    Main function to regenerate all summaries.
    """
    print("="*80)
    print("REGENERATING SUMMARIES WITH ENHANCED METRICS")
    print("="*80)
    print(f"\nThis script will regenerate summary files for all traditional methods")
    print(f"using the enhanced evaluation framework with comprehensive metrics.")

    success_count = 0
    failed_count = 0

    for method_name, (estimator_class, results_dir) in METHODS.items():
        if regenerate_summary(method_name, estimator_class, results_dir):
            success_count += 1
        else:
            failed_count += 1

    print(f"\n{'='*80}")
    print("SUMMARY REGENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n  Successfully regenerated: {success_count}/{len(METHODS)} methods")
    if failed_count > 0:
        print(f"  Failed: {failed_count} methods")

    print("\n  New summaries include:")
    print("    - MACE (Mean Average Corner Error)")
    print("    - Reprojection Error statistics")
    print("    - RMSE (Root Mean Squared Error)")
    print("    - Matching Precision/Recall/F1 Score")
    print("    - Matrix Error (Frobenius Norm)")
    print("    - Performance Metrics (Time, Memory)")
    print("    - Failure Mode Distribution")
    print("    - Robustness Metrics")
    print("="*80)


if __name__ == "__main__":
    main()

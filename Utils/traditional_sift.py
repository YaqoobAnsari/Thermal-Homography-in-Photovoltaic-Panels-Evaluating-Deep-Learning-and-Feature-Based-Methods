"""
traditional_sift.py

SIFT (Scale-Invariant Feature Transform) based homography estimation.

SIFT Features:
- Scale invariant
- Rotation invariant
- Robust to affine distortion
- Best overall accuracy but slower
- Patent-free since 2020

Author: Homography Benchmarking Project
Date: 2025
"""

import cv2
import os
import sys
from .traditional_base import TraditionalHomographyEstimator

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HOMOGRAPHY_PAIRS_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Homography Pairs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results', 'SIFT')


class SIFTHomographyEstimator(TraditionalHomographyEstimator):
    """
    SIFT-based homography estimator.
    """

    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                 edgeThreshold=10, sigma=1.6, use_flann=True,
                 ransac_reproj_threshold=5.0, ransac_method='RANSAC',
                 use_advanced_matching=True, distance_ratio=0.75):
        """
        Initialize SIFT detector.

        Args:
            nfeatures: Number of best features to retain (0 = all)
            nOctaveLayers: Number of layers in each octave
            contrastThreshold: Contrast threshold (higher = fewer features)
            edgeThreshold: Edge threshold (higher = more edge-like features)
            sigma: Gaussian sigma for initial image
            use_flann: Use FLANN matcher
            ransac_reproj_threshold: RANSAC reprojection threshold
            ransac_method: RANSAC variant ('RANSAC', 'MLESAC', 'PROSAC')
            use_advanced_matching: Use advanced FLANN matching with ratio test
            distance_ratio: Lowe's ratio test threshold
        """
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        super().__init__('SIFT', use_flann, ransac_reproj_threshold,
                        ransac_method, use_advanced_matching, distance_ratio)

    def create_detector(self):
        """
        Create SIFT detector.
        """
        return cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.nOctaveLayers,
            contrastThreshold=self.contrastThreshold,
            edgeThreshold=self.edgeThreshold,
            sigma=self.sigma
        )


def main():
    """
    Run SIFT homography estimation on all data types.
    """
    print("="*80)
    print("SIFT-BASED HOMOGRAPHY ESTIMATION")
    print("="*80)

    # Create SIFT estimator
    estimator = SIFTHomographyEstimator(
        nfeatures=0,              # Detect all features
        contrastThreshold=0.04,   # Standard threshold
        edgeThreshold=10,         # Standard threshold
        use_flann=True,           # Use FLANN for speed
        ransac_reproj_threshold=5.0
    )

    print(f"\nConfiguration:")
    print(f"  Detector: SIFT")
    print(f"  Max features: {estimator.nfeatures if estimator.nfeatures > 0 else 'unlimited'}")
    print(f"  Contrast threshold: {estimator.contrastThreshold}")
    print(f"  Edge threshold: {estimator.edgeThreshold}")
    print(f"  Matcher: {'FLANN' if estimator.use_flann else 'BFMatcher'}")
    print(f"  RANSAC threshold: {estimator.ransac_reproj_threshold} pixels")

    # Get all data type directories
    if not os.path.exists(HOMOGRAPHY_PAIRS_DIR):
        print(f"\nERROR: Homography pairs directory not found: {HOMOGRAPHY_PAIRS_DIR}")
        print("Please run generate_homography_pairs.py first.")
        return

    data_types = [d for d in os.listdir(HOMOGRAPHY_PAIRS_DIR)
                 if os.path.isdir(os.path.join(HOMOGRAPHY_PAIRS_DIR, d))]

    if len(data_types) == 0:
        print(f"\nERROR: No data types found in {HOMOGRAPHY_PAIRS_DIR}")
        return

    print(f"\nFound {len(data_types)} data types to process")

    # Process each data type
    for idx, data_type in enumerate(sorted(data_types), 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(data_types)}] Processing: {data_type}")
        print(f"{'='*80}")

        data_type_dir = os.path.join(HOMOGRAPHY_PAIRS_DIR, data_type)
        estimator.process_dataset(data_type_dir, RESULTS_DIR)

    # Save summary
    print(f"\n{'='*80}")
    print("SAVING SUMMARY")
    print(f"{'='*80}")

    summary = estimator.save_summary(RESULTS_DIR)

    print(f"\nFinal Results:")
    print(f"  Method: {summary['method']}")
    print(f"  Total pairs: {summary['total_pairs']}")
    print(f"  Successful: {summary['successful_pairs']}")
    print(f"  Failed: {summary['failed_pairs']}")
    print(f"  Success rate: {summary['success_rate']*100:.2f}%")

    if 'mean_corner_error' in summary:
        print(f"\n  Mean corner error: {summary['mean_corner_error']:.4f} pixels")
        print(f"  Median corner error: {summary['median_corner_error']:.4f} pixels")
        print(f"  Std corner error: {summary['std_corner_error']:.4f} pixels")

    if 'mean_computation_time' in summary:
        print(f"\n  Mean computation time: {summary['mean_computation_time']*1000:.2f} ms")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

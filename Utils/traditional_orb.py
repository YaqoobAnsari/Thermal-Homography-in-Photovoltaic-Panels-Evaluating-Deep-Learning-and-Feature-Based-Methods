"""
traditional_orb.py

ORB (Oriented FAST and Rotated BRIEF) based homography estimation.

ORB Features:
- Very fast (real-time capable)
- Binary descriptors (efficient matching)
- Rotation invariant
- Scale invariant (pyramid approach)
- Patent-free alternative to SIFT/SURF
- May have lower accuracy than SIFT

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results', 'ORB')


class ORBHomographyEstimator(TraditionalHomographyEstimator):
    """
    ORB-based homography estimator.
    """

    def __init__(self, nfeatures=500, scaleFactor=1.2, nlevels=8,
                 edgeThreshold=31, firstLevel=0, WTA_K=2,
                 scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                 fastThreshold=20, use_flann=True,
                 ransac_reproj_threshold=5.0, ransac_method='RANSAC',
                 use_advanced_matching=True, distance_ratio=0.75):
        """
        Initialize ORB detector.

        Args:
            nfeatures: Maximum number of features to retain
            scaleFactor: Pyramid decimation ratio (> 1)
            nlevels: Number of pyramid levels
            edgeThreshold: Border size where features are not detected
            firstLevel: Level of pyramid to put source image
            WTA_K: Number of points for oriented BRIEF descriptor
            scoreType: cv2.ORB_HARRIS_SCORE or cv2.ORB_FAST_SCORE
            patchSize: Size of patch used by oriented BRIEF descriptor
            fastThreshold: FAST threshold
            use_flann: Use FLANN matcher (LSH for binary descriptors)
            ransac_reproj_threshold: RANSAC reprojection threshold
            ransac_method: RANSAC variant ('RANSAC', 'MLESAC', 'PROSAC')
            use_advanced_matching: Use advanced FLANN matching with ratio test
            distance_ratio: Lowe's ratio test threshold
        """
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.edgeThreshold = edgeThreshold
        self.firstLevel = firstLevel
        self.WTA_K = WTA_K
        self.scoreType = scoreType
        self.patchSize = patchSize
        self.fastThreshold = fastThreshold

        super().__init__('ORB', use_flann, ransac_reproj_threshold,
                        ransac_method, use_advanced_matching, distance_ratio)

    def create_detector(self):
        """
        Create ORB detector.
        """
        return cv2.ORB_create(
            nfeatures=self.nfeatures,
            scaleFactor=self.scaleFactor,
            nlevels=self.nlevels,
            edgeThreshold=self.edgeThreshold,
            firstLevel=self.firstLevel,
            WTA_K=self.WTA_K,
            scoreType=self.scoreType,
            patchSize=self.patchSize,
            fastThreshold=self.fastThreshold
        )


def main():
    """
    Run ORB homography estimation on all data types.
    """
    print("="*80)
    print("ORB-BASED HOMOGRAPHY ESTIMATION")
    print("="*80)

    # Create ORB estimator
    estimator = ORBHomographyEstimator(
        nfeatures=1000,           # Detect up to 1000 features
        scaleFactor=1.2,          # 20% scale reduction between levels
        nlevels=8,                # 8 pyramid levels
        edgeThreshold=31,         # Standard border
        WTA_K=2,                  # 2-point BRIEF (faster)
        scoreType=cv2.ORB_HARRIS_SCORE,  # Harris corner score
        patchSize=31,             # Standard patch size
        fastThreshold=20,         # FAST threshold
        use_flann=True,           # Use FLANN with LSH
        ransac_reproj_threshold=5.0
    )

    print(f"\nConfiguration:")
    print(f"  Detector: ORB")
    print(f"  Max features: {estimator.nfeatures}")
    print(f"  Scale factor: {estimator.scaleFactor}")
    print(f"  Pyramid levels: {estimator.nlevels}")
    print(f"  Score type: {'Harris' if estimator.scoreType == cv2.ORB_HARRIS_SCORE else 'FAST'}")
    print(f"  Patch size: {estimator.patchSize}")
    print(f"  Matcher: {'FLANN-LSH' if estimator.use_flann else 'BFMatcher-Hamming'}")
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

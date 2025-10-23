"""
traditional_akaze.py

AKAZE (Accelerated-KAZE) based homography estimation.

AKAZE Features:
- Fast nonlinear scale space construction
- Binary or floating-point descriptors
- Better than SURF, faster than KAZE
- Good balance of speed and accuracy
- Robust to noise and blur
- Patent-free

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results', 'AKAZE')


class AKAZEHomographyEstimator(TraditionalHomographyEstimator):
    """
    AKAZE-based homography estimator.
    """

    def __init__(self, descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                 descriptor_size=0, descriptor_channels=3,
                 threshold=0.001, nOctaves=4, nOctaveLayers=4,
                 diffusivity=cv2.KAZE_DIFF_PM_G2, use_flann=True,
                 ransac_reproj_threshold=5.0, ransac_method='RANSAC',
                 use_advanced_matching=True, distance_ratio=0.75):
        """
        Initialize AKAZE detector.

        Args:
            descriptor_type: Type of descriptor
                - cv2.AKAZE_DESCRIPTOR_KAZE (float)
                - cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT (float, no rotation)
                - cv2.AKAZE_DESCRIPTOR_MLDB (binary, default)
                - cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT (binary, no rotation)
            descriptor_size: Size of descriptor (0 = full size)
            descriptor_channels: Number of channels in descriptor
            threshold: Detector response threshold
            nOctaves: Number of octaves
            nOctaveLayers: Number of layers per octave
            diffusivity: Diffusivity type (cv2.KAZE_DIFF_PM_G1, PM_G2, WEICKERT, CHARBONNIER)
            use_flann: Use FLANN matcher
            ransac_reproj_threshold: RANSAC reprojection threshold
            ransac_method: RANSAC variant ('RANSAC', 'MLESAC', 'PROSAC')
            use_advanced_matching: Use advanced FLANN matching with ratio test
            distance_ratio: Lowe's ratio test threshold
        """
        self.descriptor_type = descriptor_type
        self.descriptor_size = descriptor_size
        self.descriptor_channels = descriptor_channels
        self.threshold = threshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.diffusivity = diffusivity

        super().__init__('AKAZE', use_flann, ransac_reproj_threshold,
                        ransac_method, use_advanced_matching, distance_ratio)

    def create_detector(self):
        """
        Create AKAZE detector.
        """
        return cv2.AKAZE_create(
            descriptor_type=self.descriptor_type,
            descriptor_size=self.descriptor_size,
            descriptor_channels=self.descriptor_channels,
            threshold=self.threshold,
            nOctaves=self.nOctaves,
            nOctaveLayers=self.nOctaveLayers,
            diffusivity=self.diffusivity
        )


def main():
    """
    Run AKAZE homography estimation on all data types.
    """
    print("="*80)
    print("AKAZE-BASED HOMOGRAPHY ESTIMATION")
    print("="*80)

    # Create AKAZE estimator (using binary MLDB descriptor)
    estimator = AKAZEHomographyEstimator(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # Binary descriptor
        descriptor_size=0,        # Full size
        descriptor_channels=3,    # Standard
        threshold=0.001,          # Detector threshold
        nOctaves=4,               # Pyramid levels
        nOctaveLayers=4,          # Sublayers
        diffusivity=cv2.KAZE_DIFF_PM_G2,  # Perona-Malik G2
        use_flann=True,           # FLANN with LSH for binary
        ransac_reproj_threshold=5.0
    )

    descriptor_names = {
        cv2.AKAZE_DESCRIPTOR_KAZE: 'KAZE (float)',
        cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT: 'KAZE-Upright (float)',
        cv2.AKAZE_DESCRIPTOR_MLDB: 'MLDB (binary)',
        cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT: 'MLDB-Upright (binary)'
    }

    print(f"\nConfiguration:")
    print(f"  Detector: AKAZE")
    print(f"  Descriptor: {descriptor_names.get(estimator.descriptor_type, 'Unknown')}")
    print(f"  Threshold: {estimator.threshold}")
    print(f"  Octaves: {estimator.nOctaves}")
    print(f"  Layers per octave: {estimator.nOctaveLayers}")
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

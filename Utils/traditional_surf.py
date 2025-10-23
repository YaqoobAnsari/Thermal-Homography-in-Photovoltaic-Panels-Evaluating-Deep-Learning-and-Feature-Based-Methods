"""
traditional_surf.py

SURF (Speeded-Up Robust Features) based homography estimation.

SURF Features:
- Faster than SIFT
- Scale and rotation invariant
- Uses integral images for speed
- Good balance of speed and accuracy
- Patented (non-free in OpenCV, requires opencv-contrib-python)

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results', 'SURF')


class SURFHomographyEstimator(TraditionalHomographyEstimator):
    """
    SURF-based homography estimator.
    """

    def __init__(self, hessianThreshold=400, nOctaves=4, nOctaveLayers=3,
                 extended=False, upright=False, use_flann=True,
                 ransac_reproj_threshold=5.0):
        """
        Initialize SURF detector.

        Args:
            hessianThreshold: Hessian threshold (lower = more features)
            nOctaves: Number of octaves
            nOctaveLayers: Number of layers per octave
            extended: Use extended 128-element descriptors (False = 64-element)
            upright: Don't compute orientation (faster, not rotation invariant)
            use_flann: Use FLANN matcher
            ransac_reproj_threshold: RANSAC reprojection threshold
        """
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.extended = extended
        self.upright = upright

        super().__init__('SURF', use_flann, ransac_reproj_threshold)

    def create_detector(self):
        """
        Create SURF detector.
        Note: Requires opencv-contrib-python (non-free module)
        """
        try:
            return cv2.xfeatures2d.SURF_create(
                hessianThreshold=self.hessianThreshold,
                nOctaves=self.nOctaves,
                nOctaveLayers=self.nOctaveLayers,
                extended=self.extended,
                upright=self.upright
            )
        except AttributeError:
            print("\nERROR: SURF is not available.")
            print("SURF is a patented algorithm and requires opencv-contrib-python.")
            print("Install with: pip install opencv-contrib-python")
            sys.exit(1)


def main():
    """
    Run SURF homography estimation on all data types.
    """
    print("="*80)
    print("SURF-BASED HOMOGRAPHY ESTIMATION")
    print("="*80)

    # Create SURF estimator
    estimator = SURFHomographyEstimator(
        hessianThreshold=400,     # Standard threshold
        nOctaves=4,               # Standard pyramid
        nOctaveLayers=3,          # Standard layers
        extended=False,           # 64-element descriptors (faster)
        upright=False,            # Rotation invariant
        use_flann=True,           # Use FLANN for speed
        ransac_reproj_threshold=5.0
    )

    print(f"\nConfiguration:")
    print(f"  Detector: SURF")
    print(f"  Hessian threshold: {estimator.hessianThreshold}")
    print(f"  Octaves: {estimator.nOctaves}")
    print(f"  Layers per octave: {estimator.nOctaveLayers}")
    print(f"  Descriptor size: {'128' if estimator.extended else '64'}")
    print(f"  Rotation invariant: {'No' if estimator.upright else 'Yes'}")
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

"""
traditional_base.py

Base class for traditional feature-based homography estimation methods.
Provides common functionality for feature detection, matching, and evaluation.

Author: Homography Benchmarking Project
Date: 2025
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import psutil
from abc import ABC, abstractmethod

# Add code directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from .enhanced_evaluation import EnhancedHomographyEvaluator
from .ransac_variants import (
    RobustHomographyEstimator,
    RANSACConfig,
    MatchingConfig,
    FLANNMatcher
)


class TraditionalHomographyEstimator(ABC):
    """
    Abstract base class for traditional homography estimation methods.

    All traditional methods (SIFT, SURF, ORB, etc.) inherit from this class
    and implement the create_detector() method.
    """

    def __init__(
        self,
        method_name,
        use_flann=True,
        ransac_reproj_threshold=5.0,
        ransac_method='RANSAC',
        use_advanced_matching=True,
        distance_ratio=0.75
    ):
        """
        Initialize the homography estimator.

        Args:
            method_name: Name of the method (e.g., 'SIFT', 'ORB')
            use_flann: Whether to use FLANN matcher (faster) or BFMatcher
            ransac_reproj_threshold: RANSAC reprojection threshold in pixels
            ransac_method: RANSAC variant to use ('RANSAC', 'MLESAC', 'PROSAC')
            use_advanced_matching: Whether to use advanced FLANN matching with ratio test
            distance_ratio: Lowe's ratio test threshold (default 0.75)
        """
        self.method_name = method_name
        self.use_flann = use_flann
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_method = ransac_method
        self.use_advanced_matching = use_advanced_matching
        self.distance_ratio = distance_ratio

        # Create detector/descriptor extractor
        self.detector = self.create_detector()

        # Create matcher
        if use_advanced_matching:
            # Use advanced FLANN matcher with ratio test
            binary_methods = ['ORB', 'BRISK', 'AKAZE']
            descriptor_type = 'binary' if method_name in binary_methods else 'float32'

            matching_config = MatchingConfig(
                matcher_type='FLANN' if use_flann else 'BF',
                distance_ratio=distance_ratio
            )
            self.matcher = FLANNMatcher(matching_config, descriptor_type)
        else:
            # Use legacy matcher
            self.matcher = self.create_matcher()

        # RANSAC variant configuration
        ransac_config = RANSACConfig(
            method=ransac_method,
            threshold=ransac_reproj_threshold
        )
        self.robust_estimator = RobustHomographyEstimator(
            ransac_config=ransac_config,
            matching_config=MatchingConfig(
                matcher_type='FLANN' if use_flann else 'BF',
                distance_ratio=distance_ratio
            )
        )

        # Enhanced evaluator
        self.enhanced_evaluator = EnhancedHomographyEvaluator(patch_size=(256, 256))

        # Statistics
        self.stats = {
            'method': method_name,
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'errors': [],
            'computation_times': [],
            'keypoint_counts': [],
            'match_counts': [],
            'inlier_counts': [],
            'per_datatype': {}  # Track stats per data type
        }

    @abstractmethod
    def create_detector(self):
        """
        Create and return the feature detector/descriptor.
        Must be implemented by each subclass.

        Returns:
            OpenCV detector object
        """
        pass

    def create_matcher(self):
        """
        Create feature matcher based on descriptor type.

        Returns:
            OpenCV matcher object (FLANN or BFMatcher)
        """
        # Determine if descriptor is binary (ORB, BRISK, AKAZE) or float (SIFT, SURF, KAZE)
        binary_methods = ['ORB', 'BRISK', 'AKAZE']
        is_binary = self.method_name in binary_methods

        if self.use_flann and not is_binary:
            # FLANN matcher for float descriptors (SIFT, SURF, AKAZE, KAZE)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.use_flann and is_binary:
            # FLANN matcher for binary descriptors (ORB, BRISK)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute Force matcher
            if is_binary:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        return matcher

    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors.

        Args:
            image: Input image (grayscale or color)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and compute
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_threshold=None):
        """
        Match features using ratio test (Lowe's ratio test).

        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2
            ratio_threshold: Ratio test threshold (default None, uses self.distance_ratio)

        Returns:
            Tuple of (good_matches, matching_stats) if advanced matching,
            or just good_matches if legacy matching
        """
        if ratio_threshold is None:
            ratio_threshold = self.distance_ratio

        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            if self.use_advanced_matching:
                return [], {'error': 'Insufficient descriptors', 'good_matches': 0}
            return []

        # Use advanced FLANN matcher if enabled
        if self.use_advanced_matching:
            try:
                good_matches, matching_stats = self.matcher.match_with_ratio_test(desc1, desc2)
                return good_matches, matching_stats
            except Exception as e:
                print(f"    Advanced matching error: {e}")
                return [], {'error': str(e), 'good_matches': 0}

        # Legacy matching (original implementation)
        # Find k=2 nearest neighbors
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            print(f"    Matching error: {e}")
            return []

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_homography_ransac(self, kp1, kp2, matches, min_matches=4):
        """
        Estimate homography using RANSAC variant (RANSAC, MLESAC, or PROSAC).

        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: Good matches from ratio test
            min_matches: Minimum number of matches required

        Returns:
            Tuple of (H, inlier_mask, num_inliers, ransac_stats) or (None, None, 0, {}) if failed
        """
        if len(matches) < min_matches:
            return None, None, 0, {'error': 'Insufficient matches'}

        # Extract matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute quality scores for PROSAC (based on match distance)
        quality_scores = None
        if self.ransac_method == 'PROSAC':
            # Lower distance = higher quality
            distances = np.array([m.distance for m in matches])
            # Invert and normalize to [0, 1]
            quality_scores = 1.0 / (1.0 + distances)
            quality_scores = quality_scores / np.max(quality_scores) if np.max(quality_scores) > 0 else quality_scores

        # Estimate homography using robust estimator
        try:
            H, inliers, ransac_stats = self.robust_estimator.estimate_homography(
                src_pts,
                dst_pts,
                quality_scores
            )

            if H is None:
                return None, None, 0, ransac_stats

            # Convert inlier boolean array to mask format for compatibility
            mask = inliers.astype(np.uint8).reshape(-1, 1) if inliers is not None else None
            num_inliers = np.sum(inliers) if inliers is not None else 0

            return H, mask, int(num_inliers), ransac_stats

        except Exception as e:
            print(f"    Homography estimation error: {e}")
            return None, None, 0, {'error': str(e)}

    def compute_corner_error(self, H_estimated, H_ground_truth, patch_size=(256, 256)):
        """
        Compute mean corner error between estimated and ground truth homography.

        Args:
            H_estimated: Estimated homography matrix (3x3)
            H_ground_truth: Ground truth homography matrix (3x3)
            patch_size: Size of the patch (width, height)

        Returns:
            Tuple of (mean_error, max_error, corner_errors)
        """
        # Define four corners of the patch
        w, h = patch_size
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # Transform corners using both homographies
        try:
            corners_estimated = cv2.perspectiveTransform(corners, H_estimated)
            corners_ground_truth = cv2.perspectiveTransform(corners, H_ground_truth)

            # Compute Euclidean distances
            errors = np.linalg.norm(corners_estimated - corners_ground_truth, axis=2).flatten()

            mean_error = np.mean(errors)
            max_error = np.max(errors)

            return mean_error, max_error, errors

        except cv2.error as e:
            print(f"    Corner error computation failed: {e}")
            return float('inf'), float('inf'), [float('inf')] * 4

    def process_pair(self, patch_A_path, patch_B_path, H_ground_truth_path):
        """
        Process a single homography pair and compute error metrics.

        Args:
            patch_A_path: Path to patch A image
            patch_B_path: Path to patch B image
            H_ground_truth_path: Path to ground truth homography matrix

        Returns:
            Dictionary with results or None if failed
        """
        start_time = time.time()

        # Load images
        img_A = cv2.imread(patch_A_path)
        img_B = cv2.imread(patch_B_path)

        if img_A is None or img_B is None:
            return None

        # Load ground truth homography
        try:
            H_gt = np.loadtxt(H_ground_truth_path)
        except Exception as e:
            print(f"    Failed to load ground truth: {e}")
            return None

        # Detect and compute features
        kp1, desc1 = self.detect_and_compute(img_A)
        kp2, desc2 = self.detect_and_compute(img_B)

        num_kp1 = len(kp1) if kp1 is not None else 0
        num_kp2 = len(kp2) if kp2 is not None else 0

        if num_kp1 == 0 or num_kp2 == 0:
            return {
                'success': False,
                'error': 'No keypoints detected',
                'num_keypoints_A': num_kp1,
                'num_keypoints_B': num_kp2
            }

        # Match features
        matching_stats = {}
        if self.use_advanced_matching:
            matches, matching_stats = self.match_features(desc1, desc2)
        else:
            matches = self.match_features(desc1, desc2)

        num_matches = len(matches)

        if num_matches < 4:
            return {
                'success': False,
                'error': 'Insufficient matches',
                'num_keypoints_A': num_kp1,
                'num_keypoints_B': num_kp2,
                'num_matches': num_matches,
                'matching_stats': matching_stats if self.use_advanced_matching else {}
            }

        # Estimate homography using RANSAC variant
        H_est, mask, num_inliers, ransac_stats = self.estimate_homography_ransac(kp1, kp2, matches)

        if H_est is None:
            return {
                'success': False,
                'error': 'Homography estimation failed',
                'num_keypoints_A': num_kp1,
                'num_keypoints_B': num_kp2,
                'num_matches': num_matches,
                'matching_stats': matching_stats if self.use_advanced_matching else {},
                'ransac_stats': ransac_stats
            }

        # Compute errors (legacy compatibility)
        patch_size = (img_A.shape[1], img_A.shape[0])
        mean_error, max_error, corner_errors = self.compute_corner_error(
            H_est, H_gt, patch_size
        )

        computation_time = time.time() - start_time

        # Get memory usage
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)

        # Basic result dictionary
        basic_result = {
            'success': True,
            'mean_corner_error': float(mean_error),
            'max_corner_error': float(max_error),
            'corner_errors': [float(e) for e in corner_errors],
            'num_keypoints_A': int(num_kp1),
            'num_keypoints_B': int(num_kp2),
            'num_matches': int(num_matches),
            'num_inliers': int(num_inliers),
            'inlier_ratio': float(num_inliers / num_matches if num_matches > 0 else 0),
            'computation_time': float(computation_time),
            'H_estimated': [[float(v) for v in row] for row in H_est.tolist()],
            'H_ground_truth': [[float(v) for v in row] for row in H_gt.tolist()],
            'matching_stats': matching_stats if self.use_advanced_matching else {},
            'ransac_stats': ransac_stats,
            'ransac_method': self.ransac_method,
            'distance_ratio': self.distance_ratio
        }

        # Apply enhanced evaluation
        enhanced_result = self.enhanced_evaluator.evaluate_pair(
            basic_result, H_gt, computation_time, memory_usage_mb
        )

        return enhanced_result

    def process_dataset(self, data_type_dir, output_dir):
        """
        Process all homography pairs in a data type directory.

        Args:
            data_type_dir: Directory containing homography pairs for a data type
            output_dir: Output directory for results

        Returns:
            Dictionary with aggregated results
        """
        data_type_name = os.path.basename(data_type_dir)
        print(f"\n  Processing data type: {data_type_name}")

        if not os.path.exists(data_type_dir):
            print(f"    ERROR: Directory not found: {data_type_dir}")
            return None

        # Initialize per-datatype stats
        if data_type_name not in self.stats['per_datatype']:
            self.stats['per_datatype'][data_type_name] = {
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'errors': [],
                'computation_times': []
            }

        results = []

        # Iterate through all image directories
        image_dirs = [d for d in os.listdir(data_type_dir)
                     if os.path.isdir(os.path.join(data_type_dir, d))]

        for image_dir in image_dirs:
            image_path = os.path.join(data_type_dir, image_dir)

            # Iterate through all pair directories
            pair_dirs = [d for d in os.listdir(image_path)
                        if os.path.isdir(os.path.join(image_path, d)) and d.startswith('pair_')]

            for pair_dir in pair_dirs:
                pair_path = os.path.join(image_path, pair_dir)

                # Paths to required files
                patch_A = os.path.join(pair_path, 'patch_A.png')
                patch_B = os.path.join(pair_path, 'patch_B.png')
                H_gt = os.path.join(pair_path, 'homography_H.txt')

                # Check if all files exist
                if not (os.path.exists(patch_A) and os.path.exists(patch_B) and os.path.exists(H_gt)):
                    continue

                # Process pair
                result = self.process_pair(patch_A, patch_B, H_gt)

                if result is not None:
                    result['image_name'] = image_dir
                    result['pair_name'] = pair_dir
                    results.append(result)

                    # Update global stats
                    self.stats['total_pairs'] += 1
                    if result['success']:
                        self.stats['successful_pairs'] += 1
                        self.stats['errors'].append(result['mean_corner_error'])
                        self.stats['computation_times'].append(result['computation_time'])
                        self.stats['keypoint_counts'].append(result['num_keypoints_A'] + result['num_keypoints_B'])
                        self.stats['match_counts'].append(result['num_matches'])
                        self.stats['inlier_counts'].append(result['num_inliers'])
                    else:
                        self.stats['failed_pairs'] += 1

                    # Update per-datatype stats
                    self.stats['per_datatype'][data_type_name]['total_pairs'] += 1
                    if result['success']:
                        self.stats['per_datatype'][data_type_name]['successful_pairs'] += 1
                        self.stats['per_datatype'][data_type_name]['errors'].append(result['mean_corner_error'])
                        self.stats['per_datatype'][data_type_name]['computation_times'].append(result['computation_time'])
                    else:
                        self.stats['per_datatype'][data_type_name]['failed_pairs'] += 1

        # Save detailed results in organized subfolder
        os.makedirs(output_dir, exist_ok=True)
        detailed_results_dir = os.path.join(output_dir, 'detailed_results')
        os.makedirs(detailed_results_dir, exist_ok=True)

        data_type_name = os.path.basename(data_type_dir)
        results_file = os.path.join(detailed_results_dir, f'{data_type_name}_results.json')

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"    Processed {len(results)} pairs")
        print(f"    Success rate: {self.stats['successful_pairs']}/{self.stats['total_pairs']}")
        print(f"    Results saved to: {results_file}")

        return results

    def save_summary(self, output_dir):
        """
        Save summary statistics to file using enhanced evaluation metrics.
        Creates both overall summary and per-datatype summary.

        Args:
            output_dir: Output directory for summary
        """
        # Load all detailed results from detailed_results directory
        detailed_results_dir = os.path.join(output_dir, 'detailed_results')
        all_results = []

        if os.path.exists(detailed_results_dir):
            for filename in os.listdir(detailed_results_dir):
                if filename.endswith('_results.json'):
                    filepath = os.path.join(detailed_results_dir, filename)
                    with open(filepath, 'r') as f:
                        results = json.load(f)
                        all_results.extend(results)

        # Use enhanced evaluator's aggregate_statistics
        summary = self.enhanced_evaluator.aggregate_statistics(all_results)
        summary['method'] = self.method_name

        # Save overall summary
        summary_file = os.path.join(output_dir, f'{self.method_name}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Overall summary saved to: {summary_file}")

        # Per-datatype summary with enhanced metrics
        per_datatype_summary = {
            'method': self.method_name,
            'per_datatype_results': {}
        }

        best_mace = float('inf')
        best_datatype = None
        worst_mace = 0
        worst_datatype = None
        best_success = 0
        most_reliable = None

        # Group results by datatype
        if os.path.exists(detailed_results_dir):
            for filename in os.listdir(detailed_results_dir):
                if filename.endswith('_results.json'):
                    datatype = filename.replace('_results.json', '')
                    filepath = os.path.join(detailed_results_dir, filename)

                    with open(filepath, 'r') as f:
                        results = json.load(f)

                    # Use enhanced evaluator's aggregate_statistics for this datatype
                    dt_summary = self.enhanced_evaluator.aggregate_statistics(results)

                    # Track best/worst datatypes
                    if 'mace_statistics' in dt_summary and 'mean' in dt_summary['mace_statistics']:
                        mean_mace = dt_summary['mace_statistics']['mean']
                        if mean_mace < best_mace:
                            best_mace = mean_mace
                            best_datatype = datatype
                        if mean_mace > worst_mace:
                            worst_mace = mean_mace
                            worst_datatype = datatype

                    if dt_summary.get('success_rate', 0) > best_success:
                        best_success = dt_summary['success_rate']
                        most_reliable = datatype

                    per_datatype_summary['per_datatype_results'][datatype] = dt_summary

        per_datatype_summary['best_datatype'] = best_datatype
        per_datatype_summary['worst_datatype'] = worst_datatype
        per_datatype_summary['most_reliable'] = most_reliable

        # Save per-datatype summary
        per_datatype_file = os.path.join(output_dir, f'{self.method_name}_per_datatype_summary.json')
        with open(per_datatype_file, 'w') as f:
            json.dump(per_datatype_summary, f, indent=2)

        print(f"  Per-datatype summary saved to: {per_datatype_file}")

        return summary

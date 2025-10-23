"""
RANSAC Variants and Advanced Matching Strategies for Homography Estimation

This module implements:
1. MLESAC (Maximum Likelihood Estimation Sample Consensus)
2. PROSAC (Progressive Sample Consensus)
3. FLANN-based matching with distance ratio test
4. Enhanced outlier detection and robustness metrics
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class RANSACConfig:
    """Configuration for RANSAC variants."""
    method: str = 'RANSAC'  # 'RANSAC', 'MLESAC', 'PROSAC'
    max_iterations: int = 2000
    confidence: float = 0.995
    threshold: float = 3.0

    # MLESAC specific
    inlier_probability: float = 0.5

    # PROSAC specific
    use_quality_scores: bool = True
    sampling_strategy: str = 'progressive'  # 'progressive' or 'random'


@dataclass
class MatchingConfig:
    """Configuration for feature matching."""
    matcher_type: str = 'FLANN'  # 'BF' (Brute Force) or 'FLANN'
    distance_ratio: float = 0.75  # Lowe's ratio test threshold
    cross_check: bool = False

    # FLANN specific parameters
    flann_index_type: str = 'kdtree'  # 'kdtree' or 'lsh'
    flann_trees: int = 5
    flann_checks: int = 50


class FLANNMatcher:
    """FLANN-based feature matcher with distance ratio test."""

    def __init__(self, config: MatchingConfig, descriptor_type: str = 'float32'):
        """
        Initialize FLANN matcher.

        Args:
            config: Matching configuration
            descriptor_type: Type of descriptor ('float32' for SIFT/SURF/KAZE, 'binary' for ORB/BRISK/AKAZE)
        """
        self.config = config
        self.descriptor_type = descriptor_type
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        """Create appropriate FLANN or BF matcher based on configuration."""
        if self.config.matcher_type == 'FLANN':
            if self.descriptor_type == 'binary':
                # LSH for binary descriptors (ORB, BRISK, AKAZE)
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1  # 2
                )
            else:
                # KD-Tree for float descriptors (SIFT, SURF, KAZE)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(
                    algorithm=FLANN_INDEX_KDTREE,
                    trees=self.config.flann_trees
                )

            search_params = dict(checks=self.config.flann_checks)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute Force matcher
            if self.descriptor_type == 'binary':
                norm_type = cv2.NORM_HAMMING
            else:
                norm_type = cv2.NORM_L2

            return cv2.BFMatcher(norm_type, crossCheck=self.config.cross_check)

    def match_with_ratio_test(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> Tuple[List[cv2.DMatch], Dict[str, Any]]:
        """
        Match descriptors using distance ratio test (Lowe's ratio test).

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            Tuple of (good_matches, matching_stats)
        """
        start_time = time.time()

        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return [], {
                'total_matches': 0,
                'good_matches': 0,
                'ratio_test_passed': 0,
                'ratio_test_failed': 0,
                'matching_time': 0.0,
                'ratio_threshold': self.config.distance_ratio
            }

        # Get k=2 nearest neighbors for ratio test
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            print(f"FLANN matching error: {e}")
            return [], {
                'total_matches': 0,
                'good_matches': 0,
                'ratio_test_passed': 0,
                'ratio_test_failed': 0,
                'matching_time': 0.0,
                'ratio_threshold': self.config.distance_ratio,
                'error': str(e)
            }

        # Apply Lowe's ratio test
        good_matches = []
        ratio_passed = 0
        ratio_failed = 0
        distance_ratios = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                ratio = m.distance / n.distance
                distance_ratios.append(ratio)

                if ratio < self.config.distance_ratio:
                    good_matches.append(m)
                    ratio_passed += 1
                else:
                    ratio_failed += 1

        matching_time = time.time() - start_time

        stats = {
            'total_matches': len(matches),
            'good_matches': len(good_matches),
            'ratio_test_passed': ratio_passed,
            'ratio_test_failed': ratio_failed,
            'matching_time': matching_time,
            'ratio_threshold': self.config.distance_ratio,
            'mean_distance_ratio': float(np.mean(distance_ratios)) if distance_ratios else 0.0,
            'median_distance_ratio': float(np.median(distance_ratios)) if distance_ratios else 0.0
        }

        return good_matches, stats


class MLESACEstimator:
    """MLESAC (Maximum Likelihood Estimation Sample Consensus) for homography estimation."""

    def __init__(self, config: RANSACConfig):
        self.config = config

    def estimate_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Estimate homography using MLESAC.

        MLESAC improves upon RANSAC by using a maximum likelihood approach
        to score models instead of counting inliers.

        Args:
            src_pts: Source points (N, 2)
            dst_pts: Destination points (N, 2)

        Returns:
            Tuple of (homography_matrix, inlier_mask, stats)
        """
        start_time = time.time()

        if len(src_pts) < 4:
            return None, None, {'error': 'Insufficient points', 'estimation_time': 0.0}

        best_H = None
        best_inliers = None
        best_score = -np.inf
        iterations = 0

        # Standard deviation for inlier distribution (based on threshold)
        sigma = self.config.threshold / 2.0

        # Outlier probability
        outlier_prob = 1.0 - self.config.inlier_probability

        for i in range(self.config.max_iterations):
            iterations += 1

            # Sample 4 random points
            if len(src_pts) < 4:
                break

            indices = np.random.choice(len(src_pts), 4, replace=False)
            sample_src = src_pts[indices]
            sample_dst = dst_pts[indices]

            # Estimate homography from sample
            try:
                H, _ = cv2.findHomography(sample_src, sample_dst, 0)
                if H is None:
                    continue
            except:
                continue

            # Transform all source points
            src_pts_h = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
            transformed = (H @ src_pts_h.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]

            # Calculate reprojection errors
            errors = np.linalg.norm(transformed - dst_pts, axis=1)

            # MLESAC scoring: maximize log-likelihood
            # For each point: log(P_inlier * exp(-e^2/(2*sigma^2)) + P_outlier * 1/max_error)
            log_likelihood = 0
            max_error = np.max(errors) if len(errors) > 0 else 1.0

            for error in errors:
                # Inlier likelihood (Gaussian)
                inlier_likelihood = self.config.inlier_probability * np.exp(-error**2 / (2 * sigma**2))

                # Outlier likelihood (uniform)
                outlier_likelihood = outlier_prob / max_error

                # Total likelihood
                total_likelihood = inlier_likelihood + outlier_likelihood

                # Log likelihood (avoid log(0))
                if total_likelihood > 0:
                    log_likelihood += np.log(total_likelihood)

            # Update best model
            if log_likelihood > best_score:
                best_score = log_likelihood
                best_H = H
                best_inliers = errors < self.config.threshold

        # Refine homography using all inliers
        if best_inliers is not None and np.sum(best_inliers) >= 4:
            try:
                refined_H, refined_mask = cv2.findHomography(
                    src_pts[best_inliers],
                    dst_pts[best_inliers],
                    0
                )
                if refined_H is not None:
                    best_H = refined_H
            except:
                pass

        estimation_time = time.time() - start_time

        stats = {
            'method': 'MLESAC',
            'iterations': iterations,
            'estimation_time': estimation_time,
            'best_log_likelihood': float(best_score),
            'inlier_count': int(np.sum(best_inliers)) if best_inliers is not None else 0,
            'outlier_count': int(np.sum(~best_inliers)) if best_inliers is not None else 0
        }

        return best_H, best_inliers, stats


class PROSACEstimator:
    """PROSAC (Progressive Sample Consensus) for homography estimation."""

    def __init__(self, config: RANSACConfig):
        self.config = config

    def estimate_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        quality_scores: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Estimate homography using PROSAC.

        PROSAC exploits the quality ranking of correspondences to sample
        from progressively larger sets, starting with the highest quality matches.

        Args:
            src_pts: Source points (N, 2)
            dst_pts: Destination points (N, 2)
            quality_scores: Quality scores for each match (higher is better)

        Returns:
            Tuple of (homography_matrix, inlier_mask, stats)
        """
        start_time = time.time()

        if len(src_pts) < 4:
            return None, None, {'error': 'Insufficient points', 'estimation_time': 0.0}

        # Sort points by quality if provided
        if quality_scores is not None and len(quality_scores) == len(src_pts):
            sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
            src_pts = src_pts[sorted_indices]
            dst_pts = dst_pts[sorted_indices]
        else:
            # If no quality scores, use random order (degrades to RANSAC)
            sorted_indices = np.arange(len(src_pts))

        best_H = None
        best_inliers = None
        best_inlier_count = 0
        iterations = 0

        # Progressive sampling
        n_points = len(src_pts)
        n_sample = 4  # Minimum points for homography

        # Calculate number of iterations for each subset size
        subset_size = n_sample

        for i in range(self.config.max_iterations):
            iterations += 1

            # Progressively increase subset size
            if i > 0 and i % 100 == 0 and subset_size < n_points:
                subset_size = min(subset_size + 10, n_points)

            # Sample from the current subset (higher quality points)
            if subset_size >= n_sample:
                sample_indices = np.random.choice(subset_size, n_sample, replace=False)
                sample_src = src_pts[sample_indices]
                sample_dst = dst_pts[sample_indices]
            else:
                continue

            # Estimate homography from sample
            try:
                H, _ = cv2.findHomography(sample_src, sample_dst, 0)
                if H is None:
                    continue
            except:
                continue

            # Transform all source points
            src_pts_h = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
            transformed = (H @ src_pts_h.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]

            # Calculate reprojection errors
            errors = np.linalg.norm(transformed - dst_pts, axis=1)

            # Count inliers
            inliers = errors < self.config.threshold
            inlier_count = np.sum(inliers)

            # Update best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_H = H
                best_inliers = inliers

        # Refine homography using all inliers
        if best_inliers is not None and np.sum(best_inliers) >= 4:
            try:
                refined_H, refined_mask = cv2.findHomography(
                    src_pts[best_inliers],
                    dst_pts[best_inliers],
                    0
                )
                if refined_H is not None:
                    best_H = refined_H
            except:
                pass

        estimation_time = time.time() - start_time

        stats = {
            'method': 'PROSAC',
            'iterations': iterations,
            'estimation_time': estimation_time,
            'final_subset_size': subset_size,
            'inlier_count': int(best_inlier_count),
            'outlier_count': int(n_points - best_inlier_count)
        }

        return best_H, best_inliers, stats


class RobustHomographyEstimator:
    """Unified interface for different RANSAC variants."""

    def __init__(self, ransac_config: RANSACConfig = None, matching_config: MatchingConfig = None):
        """
        Initialize robust homography estimator.

        Args:
            ransac_config: Configuration for RANSAC variant
            matching_config: Configuration for feature matching
        """
        self.ransac_config = ransac_config or RANSACConfig()
        self.matching_config = matching_config or MatchingConfig()

        # Initialize estimators
        if self.ransac_config.method == 'MLESAC':
            self.ransac_estimator = MLESACEstimator(self.ransac_config)
        elif self.ransac_config.method == 'PROSAC':
            self.ransac_estimator = PROSACEstimator(self.ransac_config)
        else:
            self.ransac_estimator = None  # Use OpenCV's RANSAC

    def create_matcher(self, descriptor_type: str = 'float32') -> FLANNMatcher:
        """Create a FLANN matcher with the current configuration."""
        return FLANNMatcher(self.matching_config, descriptor_type)

    def estimate_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        quality_scores: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Estimate homography using the configured RANSAC variant.

        Args:
            src_pts: Source points (N, 2)
            dst_pts: Destination points (N, 2)
            quality_scores: Quality scores for matches (for PROSAC)

        Returns:
            Tuple of (homography_matrix, inlier_mask, stats)
        """
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, None, {
                'error': 'Insufficient points',
                'method': self.ransac_config.method,
                'estimation_time': 0.0
            }

        if self.ransac_config.method == 'RANSAC':
            # Use OpenCV's RANSAC
            start_time = time.time()
            try:
                H, mask = cv2.findHomography(
                    src_pts,
                    dst_pts,
                    cv2.RANSAC,
                    self.ransac_config.threshold
                )
                estimation_time = time.time() - start_time

                inliers = mask.ravel() == 1 if mask is not None else np.zeros(len(src_pts), dtype=bool)

                stats = {
                    'method': 'RANSAC',
                    'estimation_time': estimation_time,
                    'inlier_count': int(np.sum(inliers)),
                    'outlier_count': int(np.sum(~inliers))
                }

                return H, inliers, stats
            except Exception as e:
                return None, None, {'error': str(e), 'method': 'RANSAC', 'estimation_time': 0.0}

        elif self.ransac_config.method == 'MLESAC':
            return self.ransac_estimator.estimate_homography(src_pts, dst_pts)

        elif self.ransac_config.method == 'PROSAC':
            return self.ransac_estimator.estimate_homography(src_pts, dst_pts, quality_scores)

        else:
            return None, None, {'error': f'Unknown RANSAC method: {self.ransac_config.method}'}


# Convenience functions
def estimate_homography_with_ransac_variant(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    method: str = 'RANSAC',
    threshold: float = 3.0,
    quality_scores: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Convenience function to estimate homography with different RANSAC variants.

    Args:
        src_pts: Source points
        dst_pts: Destination points
        method: RANSAC variant ('RANSAC', 'MLESAC', 'PROSAC')
        threshold: Inlier threshold
        quality_scores: Quality scores for PROSAC

    Returns:
        Tuple of (homography_matrix, inlier_mask, stats)
    """
    config = RANSACConfig(method=method, threshold=threshold)
    estimator = RobustHomographyEstimator(ransac_config=config)
    return estimator.estimate_homography(src_pts, dst_pts, quality_scores)


def match_features_with_flann(
    desc1: np.ndarray,
    desc2: np.ndarray,
    descriptor_type: str = 'float32',
    distance_ratio: float = 0.75
) -> Tuple[List[cv2.DMatch], Dict[str, Any]]:
    """
    Convenience function to match features using FLANN with ratio test.

    Args:
        desc1: First set of descriptors
        desc2: Second set of descriptors
        descriptor_type: Type of descriptor ('float32' or 'binary')
        distance_ratio: Lowe's ratio test threshold

    Returns:
        Tuple of (good_matches, matching_stats)
    """
    config = MatchingConfig(matcher_type='FLANN', distance_ratio=distance_ratio)
    matcher = FLANNMatcher(config, descriptor_type)
    return matcher.match_with_ratio_test(desc1, desc2)

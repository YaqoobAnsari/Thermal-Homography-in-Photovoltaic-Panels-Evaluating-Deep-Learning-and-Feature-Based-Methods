"""
enhanced_evaluation.py

Comprehensive evaluation framework with advanced metrics for homography estimation.
Implements MACE, Reprojection Error, RMSE, Precision/Recall, and Failure Analysis.

Author: Homography Benchmarking Project
Date: 2025
"""

import cv2
import numpy as np
import json
import os
import time
import psutil
from collections import defaultdict


class EnhancedHomographyEvaluator:
    """
    Enhanced evaluation framework for homography estimation with comprehensive metrics.
    """

    def __init__(self, patch_size=(256, 256)):
        """
        Initialize the evaluator.

        Args:
            patch_size: Size of the patches (width, height)
        """
        self.patch_size = patch_size
        self.results = []

        # Define corner points for evaluation
        w, h = patch_size
        self.corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # Additional evaluation points (center and edge midpoints)
        self.eval_points = np.float32([
            [0, 0], [w, 0], [w, h], [0, h],  # Corners
            [w/2, 0], [w, h/2], [w/2, h], [0, h/2],  # Edge midpoints
            [w/2, h/2]  # Center
        ]).reshape(-1, 1, 2)

    def compute_mace(self, H_estimated, H_ground_truth):
        """
        Compute Mean Average Corner Error (MACE).

        Args:
            H_estimated: Estimated homography matrix (3x3)
            H_ground_truth: Ground truth homography matrix (3x3)

        Returns:
            Dictionary with MACE and per-corner errors
        """
        try:
            # Transform corners using both homographies
            corners_estimated = cv2.perspectiveTransform(self.corners, H_estimated)
            corners_ground_truth = cv2.perspectiveTransform(self.corners, H_ground_truth)

            # Compute Euclidean distances for each corner
            corner_errors = np.linalg.norm(
                corners_estimated - corners_ground_truth, axis=2
            ).flatten()

            mace = np.mean(corner_errors)

            return {
                'mace': float(mace),
                'corner_errors': [float(e) for e in corner_errors],
                'max_corner_error': float(np.max(corner_errors)),
                'min_corner_error': float(np.min(corner_errors)),
                'std_corner_error': float(np.std(corner_errors))
            }
        except:
            return {
                'mace': float('inf'),
                'corner_errors': [float('inf')] * 4,
                'max_corner_error': float('inf'),
                'min_corner_error': float('inf'),
                'std_corner_error': float('inf')
            }

    def compute_reprojection_error(self, H_estimated, H_ground_truth):
        """
        Compute Reprojection Error across multiple points.

        Args:
            H_estimated: Estimated homography matrix (3x3)
            H_ground_truth: Ground truth homography matrix (3x3)

        Returns:
            Dictionary with reprojection error metrics
        """
        try:
            # Transform evaluation points
            points_estimated = cv2.perspectiveTransform(self.eval_points, H_estimated)
            points_ground_truth = cv2.perspectiveTransform(self.eval_points, H_ground_truth)

            # Compute errors for all points
            errors = np.linalg.norm(
                points_estimated - points_ground_truth, axis=2
            ).flatten()

            return {
                'mean_reprojection_error': float(np.mean(errors)),
                'median_reprojection_error': float(np.median(errors)),
                'max_reprojection_error': float(np.max(errors)),
                'std_reprojection_error': float(np.std(errors))
            }
        except:
            return {
                'mean_reprojection_error': float('inf'),
                'median_reprojection_error': float('inf'),
                'max_reprojection_error': float('inf'),
                'std_reprojection_error': float('inf')
            }

    def compute_rmse(self, H_estimated, H_ground_truth):
        """
        Compute Root Mean Squared Error (RMSE) across evaluation points.

        Args:
            H_estimated: Estimated homography matrix (3x3)
            H_ground_truth: Ground truth homography matrix (3x3)

        Returns:
            Float RMSE value
        """
        try:
            points_estimated = cv2.perspectiveTransform(self.eval_points, H_estimated)
            points_ground_truth = cv2.perspectiveTransform(self.eval_points, H_ground_truth)

            squared_errors = np.sum((points_estimated - points_ground_truth) ** 2, axis=2).flatten()
            rmse = np.sqrt(np.mean(squared_errors))

            return float(rmse)
        except:
            return float('inf')

    def compute_matching_precision_recall(self, num_matches, num_inliers,
                                          num_keypoints_A, num_keypoints_B):
        """
        Compute matching precision and recall metrics.

        Args:
            num_matches: Number of feature matches
            num_inliers: Number of RANSAC inliers
            num_keypoints_A: Number of keypoints in image A
            num_keypoints_B: Number of keypoints in image B

        Returns:
            Dictionary with precision and recall metrics
        """
        # Precision: ratio of inliers to matches
        precision = num_inliers / num_matches if num_matches > 0 else 0.0

        # Recall: ratio of matches to minimum keypoints
        # (how many of the available keypoints were matched)
        min_keypoints = min(num_keypoints_A, num_keypoints_B)
        recall = num_matches / min_keypoints if min_keypoints > 0 else 0.0

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'matching_precision': float(precision),
            'matching_recall': float(recall),
            'f1_score': float(f1_score),
            'inlier_ratio': float(num_inliers / num_matches if num_matches > 0 else 0)
        }

    def compute_homography_matrix_error(self, H_estimated, H_ground_truth):
        """
        Compute direct matrix error between estimated and ground truth homography.

        Args:
            H_estimated: Estimated homography matrix (3x3)
            H_ground_truth: Ground truth homography matrix (3x3)

        Returns:
            Dictionary with matrix error metrics
        """
        try:
            # Normalize both matrices by bottom-right element
            # Check for zero or near-zero values to avoid division errors
            if abs(H_estimated[2, 2]) < 1e-10 or abs(H_ground_truth[2, 2]) < 1e-10:
                return {
                    'frobenius_norm': float('inf'),
                    'max_element_error': float('inf'),
                    'mean_element_error': float('inf')
                }

            H_est_norm = H_estimated / H_estimated[2, 2]
            H_gt_norm = H_ground_truth / H_ground_truth[2, 2]

            # Frobenius norm of difference
            frobenius_norm = np.linalg.norm(H_est_norm - H_gt_norm, 'fro')

            # Element-wise differences
            element_diff = np.abs(H_est_norm - H_gt_norm)

            return {
                'frobenius_norm': float(frobenius_norm),
                'max_element_error': float(np.max(element_diff)),
                'mean_element_error': float(np.mean(element_diff))
            }
        except:
            return {
                'frobenius_norm': float('inf'),
                'max_element_error': float('inf'),
                'mean_element_error': float('inf')
            }

    def analyze_failure_mode(self, result):
        """
        Analyze and categorize failure modes.

        Args:
            result: Dictionary with evaluation results

        Returns:
            String describing failure mode
        """
        if result.get('success', False):
            # Check for high error despite success
            if result.get('mace', {}).get('mace', 0) > 10.0:
                return 'high_error_estimation'
            return 'success'

        error = result.get('error', 'unknown')

        # Categorize failures
        if 'keypoints' in error.lower():
            return 'insufficient_keypoints'
        elif 'matches' in error.lower():
            return 'insufficient_matches'
        elif 'homography' in error.lower():
            return 'homography_estimation_failed'
        else:
            return 'unknown_failure'

    def evaluate_pair(self, result_dict, H_ground_truth, computation_time,
                     memory_usage_mb=None):
        """
        Perform comprehensive evaluation on a single pair result.

        Args:
            result_dict: Dictionary from traditional method processing
            H_ground_truth: Ground truth homography matrix
            computation_time: Time taken for computation
            memory_usage_mb: Memory usage in MB (optional)

        Returns:
            Enhanced result dictionary with all metrics
        """
        enhanced_result = result_dict.copy()

        if not result_dict.get('success', False):
            # For failures, add basic metrics
            enhanced_result['mace'] = {'mace': float('inf')}
            enhanced_result['reprojection_error'] = {'mean_reprojection_error': float('inf')}
            enhanced_result['rmse'] = float('inf')
            enhanced_result['matching_metrics'] = {
                'matching_precision': 0.0,
                'matching_recall': 0.0,
                'f1_score': 0.0,
                'inlier_ratio': 0.0
            }
            enhanced_result['matrix_error'] = {
                'frobenius_norm': float('inf'),
                'max_element_error': float('inf'),
                'mean_element_error': float('inf')
            }
        else:
            # Compute all metrics
            H_estimated = np.array(result_dict['H_estimated'])
            H_gt = np.array(result_dict['H_ground_truth'])

            # MACE
            enhanced_result['mace'] = self.compute_mace(H_estimated, H_gt)

            # Reprojection Error
            enhanced_result['reprojection_error'] = self.compute_reprojection_error(
                H_estimated, H_gt
            )

            # RMSE
            enhanced_result['rmse'] = self.compute_rmse(H_estimated, H_gt)

            # Matching Precision/Recall
            enhanced_result['matching_metrics'] = self.compute_matching_precision_recall(
                result_dict.get('num_matches', 0),
                result_dict.get('num_inliers', 0),
                result_dict.get('num_keypoints_A', 0),
                result_dict.get('num_keypoints_B', 0)
            )

            # Matrix Error
            enhanced_result['matrix_error'] = self.compute_homography_matrix_error(
                H_estimated, H_gt
            )

        # Performance metrics
        enhanced_result['performance'] = {
            'computation_time': float(computation_time),
            'memory_usage_mb': float(memory_usage_mb) if memory_usage_mb else None
        }

        # Failure mode analysis
        enhanced_result['failure_mode'] = self.analyze_failure_mode(enhanced_result)

        return enhanced_result

    def aggregate_statistics(self, enhanced_results):
        """
        Compute aggregate statistics across all results.

        Args:
            enhanced_results: List of enhanced result dictionaries

        Returns:
            Dictionary with aggregate statistics
        """
        successful = [r for r in enhanced_results if r.get('success', False)]
        failed = [r for r in enhanced_results if not r.get('success', False)]

        stats = {
            'total_pairs': len(enhanced_results),
            'successful_pairs': len(successful),
            'failed_pairs': len(failed),
            'success_rate': len(successful) / len(enhanced_results) if enhanced_results else 0.0
        }

        if successful:
            # MACE statistics
            mace_values = [r['mace']['mace'] for r in successful if r['mace']['mace'] != float('inf')]
            if mace_values:
                stats['mace_statistics'] = {
                    'mean': float(np.mean(mace_values)),
                    'median': float(np.median(mace_values)),
                    'std': float(np.std(mace_values)),
                    'min': float(np.min(mace_values)),
                    'max': float(np.max(mace_values)),
                    'percentile_25': float(np.percentile(mace_values, 25)),
                    'percentile_75': float(np.percentile(mace_values, 75)),
                    'percentile_95': float(np.percentile(mace_values, 95))
                }

            # Reprojection Error statistics
            reproj_values = [r['reprojection_error']['mean_reprojection_error']
                           for r in successful
                           if r['reprojection_error']['mean_reprojection_error'] != float('inf')]
            if reproj_values:
                stats['reprojection_error_statistics'] = {
                    'mean': float(np.mean(reproj_values)),
                    'median': float(np.median(reproj_values)),
                    'std': float(np.std(reproj_values))
                }

            # RMSE statistics
            rmse_values = [r['rmse'] for r in successful if r['rmse'] != float('inf')]
            if rmse_values:
                stats['rmse_statistics'] = {
                    'mean': float(np.mean(rmse_values)),
                    'median': float(np.median(rmse_values)),
                    'std': float(np.std(rmse_values))
                }

            # Matching statistics
            precision_values = [r['matching_metrics']['matching_precision'] for r in successful]
            recall_values = [r['matching_metrics']['matching_recall'] for r in successful]
            f1_values = [r['matching_metrics']['f1_score'] for r in successful]

            stats['matching_statistics'] = {
                'mean_precision': float(np.mean(precision_values)),
                'mean_recall': float(np.mean(recall_values)),
                'mean_f1_score': float(np.mean(f1_values))
            }

            # Performance statistics
            comp_times = [r['performance']['computation_time'] for r in successful]
            stats['performance_statistics'] = {
                'mean_computation_time': float(np.mean(comp_times)),
                'median_computation_time': float(np.median(comp_times)),
                'total_computation_time': float(np.sum(comp_times))
            }

            # Matrix error statistics
            frob_norms = [r['matrix_error']['frobenius_norm']
                         for r in successful
                         if r['matrix_error']['frobenius_norm'] != float('inf')]
            if frob_norms:
                stats['matrix_error_statistics'] = {
                    'mean_frobenius_norm': float(np.mean(frob_norms)),
                    'median_frobenius_norm': float(np.median(frob_norms))
                }

        # Failure mode analysis
        failure_modes = defaultdict(int)
        for r in enhanced_results:
            failure_mode = r.get('failure_mode', 'unknown')
            failure_modes[failure_mode] += 1

        stats['failure_mode_distribution'] = dict(failure_modes)

        # Robustness metrics
        if enhanced_results:
            stats['robustness_metrics'] = {
                'error_variance': float(np.var(mace_values)) if 'mace_values' in locals() and mace_values else None,
                'consistency_score': len(successful) / len(enhanced_results),
                'outlier_ratio': len([r for r in successful if r['mace']['mace'] > 10.0]) / len(successful) if successful else 0.0
            }

        return stats


def main():
    """
    Demonstrate usage of enhanced evaluation framework.
    """
    print("="*80)
    print("ENHANCED EVALUATION FRAMEWORK")
    print("="*80)
    print("\nThis module provides comprehensive homography evaluation metrics:")
    print("  - MACE (Mean Average Corner Error)")
    print("  - Reprojection Error")
    print("  - RMSE (Root Mean Squared Error)")
    print("  - Matching Precision & Recall")
    print("  - Matrix Error (Frobenius Norm)")
    print("  - Performance Metrics (Time, Memory)")
    print("  - Failure Mode Analysis")
    print("  - Robustness Metrics")
    print("\nImport this module in traditional_base.py for enhanced evaluation.")
    print("="*80)


if __name__ == "__main__":
    main()

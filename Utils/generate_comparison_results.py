"""
generate_comparison_results.py

Fixed version that handles RANSAC variant directories dynamically.

Generates comprehensive cross-method comparison results including:
1. Cross-method comparison JSON
2. Performance matrix CSV
3. Per-datatype comparison JSON
4. Colormap/category analysis JSON
5. RANSAC variant comparison JSON

Author: Homography Benchmarking Project
Date: 2025
"""

import os
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict


# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results')


def get_all_method_configs():
    """
    Dynamically discover all method configurations from results directory.
    Returns list of method directory names (e.g., ['SIFT_RANSAC', 'SIFT_MLESAC', ...])
    """
    if not os.path.exists(RESULTS_DIR):
        return []

    configs = []
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if os.path.isdir(item_path):
            # Check if it's a method directory (has a summary file)
            summary_file = os.path.join(item_path, f'{item}_summary.json')
            if os.path.exists(summary_file):
                configs.append(item)

    return sorted(configs)


# Colormap categories for analysis
COLORMAP_CATEGORIES = {
    'grayscale': {
        'colormaps': ['greys', 'preprocessed', 'raw'],
        'description': 'Single-channel intensity mapping preserving original thermal data',
        'characteristics': 'Monotonic luminance, maximum feature preservation',
        'pros': 'Preserves original intensity relationships, no color artifacts, optimal for feature detection',
        'cons': 'Limited perceptual range, harder to distinguish subtle temperature differences',
        'expected_performance': 'Best - maintains original gradient information crucial for feature matching'
    },
    'perceptual_uniform': {
        'colormaps': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
        'description': 'Perceptually uniform colormaps designed for consistent visual interpretation',
        'characteristics': 'Smooth gradients, perceptually linear, colorblind-friendly',
        'pros': 'Beautiful visualization, perceptually uniform, scientifically sound for human interpretation',
        'cons': 'Smooths gradients too much, reduces sharp edges, diminishes feature distinctiveness',
        'expected_performance': 'Poor - perceptual uniformity eliminates sharp transitions needed for feature detection'
    },
    'sequential_single_hue': {
        'colormaps': ['blues', 'reds', 'oranges', 'copper'],
        'description': 'Single-hue sequential colormaps with monotonic luminance',
        'characteristics': 'One primary hue, varying luminance and saturation',
        'pros': 'Maintains some gradient sharpness, intuitive temperature mapping',
        'cons': 'Limited color range, potential loss of detail in extreme values',
        'expected_performance': 'Moderate - better than perceptual but loses some feature definition'
    },
    'sequential_multi_hue': {
        'colormaps': ['hot', 'cool', 'terrain', 'nipy_spectral'],
        'description': 'Multi-hue sequential colormaps spanning color spectrum',
        'characteristics': 'Multiple hues, varying luminance, wide perceptual range',
        'pros': 'High dynamic range, distinct color transitions, good for temperature gradients',
        'cons': 'Non-monotonic luminance can confuse feature detectors',
        'expected_performance': 'Variable - depends on luminance profile and hue transitions'
    },
    'diverging': {
        'colormaps': ['coolwarm', 'seismic', 'rdbu', 'brbg', 'piyg'],
        'description': 'Diverging colormaps with central neutral point and contrasting extremes',
        'characteristics': 'Two contrasting hues, light middle, emphasizes deviations from center',
        'pros': 'Strong contrast at extremes, emphasizes hot/cold boundaries, sharp transitions',
        'cons': 'Middle values may lose definition, bimodal emphasis',
        'expected_performance': 'Good - high contrast aids feature detection, especially at boundaries'
    },
    'cyclic': {
        'colormaps': ['twilight', 'twilight_shifted', 'hsv'],
        'description': 'Cyclic colormaps for periodic data or angular measurements',
        'characteristics': 'Seamless wraparound, suitable for periodic phenomena',
        'pros': 'HSV provides maximum color variation, twilight has perceptual balance',
        'cons': 'May create artificial boundaries, HSV has luminance variations',
        'expected_performance': 'Variable - HSV better due to color diversity, twilight may smooth features'
    },
    'qualitative': {
        'colormaps': ['set1', 'set2', 'pastel1', 'tab10'],
        'description': 'Qualitative colormaps for categorical/discrete data',
        'characteristics': 'Distinct colors, not ordered, designed for categories not continuous data',
        'pros': 'Maximum color distinction, each value highly distinguishable',
        'cons': 'Not designed for continuous thermal data, creates artificial segmentation',
        'expected_performance': 'Good - high contrast between values creates strong feature boundaries'
    },
    'miscellaneous': {
        'colormaps': ['flag'],
        'description': 'Special-purpose colormaps with unique characteristics',
        'characteristics': 'High contrast, discrete steps, unconventional color schemes',
        'pros': 'Maximum visual distinction, creates sharp boundaries',
        'cons': 'Arbitrary color mapping, may not represent thermal data meaningfully',
        'expected_performance': 'Unpredictable - high contrast helps but arbitrary mapping may confuse'
    }
}


def load_method_summary(method_name):
    """Load overall summary for a method."""
    summary_file = os.path.join(RESULTS_DIR, method_name, f'{method_name}_summary.json')

    if not os.path.exists(summary_file):
        print(f"  WARNING: Summary not found for {method_name}")
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)


def load_method_per_datatype(method_name):
    """Load per-datatype summary for a method."""
    per_datatype_file = os.path.join(RESULTS_DIR, method_name, f'{method_name}_per_datatype_summary.json')

    if not os.path.exists(per_datatype_file):
        print(f"  WARNING: Per-datatype summary not found for {method_name}")
        return None

    with open(per_datatype_file, 'r') as f:
        return json.load(f)


def get_colormap_category(datatype):
    """Get the category of a colormap."""
    datatype_lower = datatype.lower().replace('colormap_', '')

    for category, info in COLORMAP_CATEGORIES.items():
        if any(cm in datatype_lower for cm in info['colormaps']):
            return category

    print(f"WARNING: Uncategorized colormap detected: {datatype}")
    return 'miscellaneous'


def generate_cross_method_comparison():
    """Generate cross-method comparison JSON with rankings and analysis."""
    print("\n" + "="*80)
    print("GENERATING CROSS-METHOD COMPARISON")
    print("="*80)

    # Get all method configurations dynamically
    methods = get_all_method_configs()

    if not methods:
        print("  ERROR: No method results found!")
        return None

    print(f"  Found {len(methods)} method configurations: {', '.join(methods)}")

    comparison = {
        'methods_compared': methods,
        'methods': {},
        'rankings': {},
        'analysis': {}
    }

    # Load all summaries
    summaries = {}
    for method in methods:
        summary = load_method_summary(method)
        if summary:
            summaries[method] = summary
            comparison['methods'][method] = summary.copy()
            comparison['methods'][method]['method'] = method

    if not summaries:
        print("  ERROR: No valid summaries loaded!")
        return None

    # Rankings by different criteria
    # By success rate
    success_rates = {m: s.get('success_rate', 0) for m, s in summaries.items()}
    ranked_by_success = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_success_rate'] = [
        {'method': m, 'success_rate': float(r)} for m, r in ranked_by_success
    ]

    # By mean MACE (lower is better)
    mean_mace = {
        m: s.get('mace_statistics', {}).get('mean', float('inf'))
        for m, s in summaries.items()
    }
    ranked_by_mace = sorted(mean_mace.items(), key=lambda x: x[1])
    comparison['rankings']['by_mean_mace'] = [
        {'method': m, 'mean_mace': float(e) if e != float('inf') else None}
        for m, e in ranked_by_mace
    ]

    # By computation time (lower is better)
    mean_times = {
        m: s.get('performance_statistics', {}).get('mean_computation_time', float('inf'))
        for m, s in summaries.items()
    }
    ranked_by_time = sorted(mean_times.items(), key=lambda x: x[1])
    comparison['rankings']['by_computation_time'] = [
        {'method': m, 'mean_time': float(t) if t != float('inf') else None}
        for m, t in ranked_by_time
    ]

    # By precision
    mean_precision = {
        m: s.get('matching_statistics', {}).get('mean_precision', 0)
        for m, s in summaries.items()
    }
    ranked_by_precision = sorted(mean_precision.items(), key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_precision'] = [
        {'method': m, 'mean_precision': float(p)} for m, p in ranked_by_precision
    ]

    # By F1 score
    mean_f1 = {
        m: s.get('matching_statistics', {}).get('mean_f1_score', 0)
        for m, s in summaries.items()
    }
    ranked_by_f1 = sorted(mean_f1.items(), key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_f1_score'] = [
        {'method': m, 'mean_f1': float(f)} for m, f in ranked_by_f1
    ]

    # Overall analysis
    comparison['analysis'] = {
        'most_accurate_mace': ranked_by_mace[0][0] if ranked_by_mace else None,
        'most_reliable': ranked_by_success[0][0] if ranked_by_success else None,
        'fastest': ranked_by_time[0][0] if ranked_by_time else None,
        'best_precision': ranked_by_precision[0][0] if ranked_by_precision else None,
        'best_f1_score': ranked_by_f1[0][0] if ranked_by_f1 else None,
        'best_overall': ranked_by_success[0][0] if ranked_by_success else None
    }

    # Save comparison
    output_file = os.path.join(RESULTS_DIR, 'cross_method_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Cross-method comparison saved to: {output_file}")
    print(f"  Compared {len(summaries)} method configurations")

    return comparison


def generate_performance_matrix_csv():
    """Generate performance matrix CSV with all methods and metrics."""
    print("\n" + "="*80)
    print("GENERATING PERFORMANCE MATRIX CSV")
    print("="*80)

    # Get all method configurations dynamically
    methods = get_all_method_configs()

    if not methods:
        print("  ERROR: No method results found!")
        return None

    # Metrics to include
    metrics = [
        'Total Pairs',
        'Successful Pairs',
        'Failed Pairs',
        'Success Rate (%)',
        'Mean MACE (px)',
        'Median MACE (px)',
        'Std MACE (px)',
        'P25 MACE (px)',
        'P75 MACE (px)',
        'P95 MACE (px)',
        'Mean Reprojection Error (px)',
        'Mean RMSE (px)',
        'Mean Precision',
        'Mean Recall',
        'Mean F1 Score',
        'Mean Time (s)',
        'Total Time (s)',
        'Mean Frobenius Norm',
        'Error Variance',
        'Consistency Score'
    ]

    # Create matrix
    matrix = []

    # Header row
    header = ['Metric'] + methods
    matrix.append(header)

    # Load all summaries
    summaries = {}
    for method in methods:
        summary = load_method_summary(method)
        if summary:
            summaries[method] = summary

    # Add each metric row
    for metric in metrics:
        row = [metric]

        for method in methods:
            if method not in summaries:
                row.append('N/A')
                continue

            summary = summaries[method]

            # Extract value based on metric
            if metric == 'Total Pairs':
                value = summary.get('total_pairs', 'N/A')
            elif metric == 'Successful Pairs':
                value = summary.get('successful_pairs', 'N/A')
            elif metric == 'Failed Pairs':
                value = summary.get('failed_pairs', 'N/A')
            elif metric == 'Success Rate (%)':
                value = f"{summary.get('success_rate', 0) * 100:.2f}"
            elif metric == 'Mean MACE (px)':
                value = summary.get('mace_statistics', {}).get('mean', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Median MACE (px)':
                value = summary.get('mace_statistics', {}).get('median', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Std MACE (px)':
                value = summary.get('mace_statistics', {}).get('std', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'P25 MACE (px)':
                value = summary.get('mace_statistics', {}).get('percentile_25', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'P75 MACE (px)':
                value = summary.get('mace_statistics', {}).get('percentile_75', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'P95 MACE (px)':
                value = summary.get('mace_statistics', {}).get('percentile_95', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Mean Reprojection Error (px)':
                value = summary.get('reprojection_error_statistics', {}).get('mean', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Mean RMSE (px)':
                value = summary.get('rmse_statistics', {}).get('mean', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Mean Precision':
                value = summary.get('matching_statistics', {}).get('mean_precision', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Mean Recall':
                value = summary.get('matching_statistics', {}).get('mean_recall', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Mean F1 Score':
                value = summary.get('matching_statistics', {}).get('mean_f1_score', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Mean Time (s)':
                value = summary.get('performance_statistics', {}).get('mean_computation_time', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Total Time (s)':
                value = summary.get('performance_statistics', {}).get('total_computation_time', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.2f}"
            elif metric == 'Mean Frobenius Norm':
                value = summary.get('matrix_error_statistics', {}).get('mean_frobenius_norm', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            elif metric == 'Error Variance':
                value = summary.get('robustness_metrics', {}).get('error_variance', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.2f}"
            elif metric == 'Consistency Score':
                value = summary.get('robustness_metrics', {}).get('consistency_score', 'N/A')
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
            else:
                value = 'N/A'

            row.append(value)

        matrix.append(row)

    # Save CSV
    output_file = os.path.join(RESULTS_DIR, 'performance_matrix.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matrix)

    print(f"\n  Performance matrix saved to: {output_file}")
    print(f"  Includes {len(methods)} method configurations and {len(metrics)} metrics")

    return matrix


def generate_per_datatype_comparison():
    """Generate per-datatype comparison JSON."""
    print("\n" + "="*80)
    print("GENERATING PER-DATATYPE COMPARISON")
    print("="*80)

    # Get all method configurations dynamically
    methods = get_all_method_configs()

    if not methods:
        print("  ERROR: No method results found!")
        return None

    # Load all per-datatype summaries
    per_datatype_data = {}
    for method in methods:
        data = load_method_per_datatype(method)
        if data:
            per_datatype_data[method] = data

    if not per_datatype_data:
        print("  ERROR: No per-datatype data found")
        return None

    # Collect all data types
    all_datatypes = set()
    for method_data in per_datatype_data.values():
        all_datatypes.update(method_data.get('per_datatype_results', {}).keys())

    all_datatypes = sorted(all_datatypes)

    # Build comparison structure
    comparison = {
        'methods': methods,
        'data_types': all_datatypes,
        'per_datatype_comparison': {},
        'best_performers': {}
    }

    # For each data type, compare all methods
    for datatype in all_datatypes:
        dt_comparison = {}

        for method in methods:
            if method not in per_datatype_data:
                continue

            dt_results = per_datatype_data[method].get('per_datatype_results', {}).get(datatype, {})

            dt_comparison[method] = {
                'total_pairs': dt_results.get('total_pairs', 0),
                'successful_pairs': dt_results.get('successful_pairs', 0),
                'success_rate': dt_results.get('success_rate', 0),
                'mean_mace': dt_results.get('mace_statistics', {}).get('mean'),
                'median_mace': dt_results.get('mace_statistics', {}).get('median'),
                'mean_time': dt_results.get('performance_statistics', {}).get('mean_computation_time'),
                'mean_precision': dt_results.get('matching_statistics', {}).get('mean_precision'),
                'mean_recall': dt_results.get('matching_statistics', {}).get('mean_recall'),
                'mean_f1_score': dt_results.get('matching_statistics', {}).get('mean_f1_score')
            }

        comparison['per_datatype_comparison'][datatype] = dt_comparison

        # Find best performer for this datatype
        valid_methods = {
            m: data for m, data in dt_comparison.items()
            if data['success_rate'] > 0
        }

        if valid_methods:
            best_by_success = max(valid_methods.items(), key=lambda x: x[1]['success_rate'])

            best_by_mace = min(
                [(m, d) for m, d in valid_methods.items() if d['mean_mace'] is not None],
                key=lambda x: x[1]['mean_mace'],
                default=(None, None)
            )

            best_by_speed = min(
                [(m, d) for m, d in valid_methods.items() if d['mean_time'] is not None],
                key=lambda x: x[1]['mean_time'],
                default=(None, None)
            )

            comparison['best_performers'][datatype] = {
                'best_success_rate': best_by_success[0],
                'best_accuracy': best_by_mace[0] if best_by_mace[0] else None,
                'fastest': best_by_speed[0] if best_by_speed[0] else None
            }

    # Save comparison
    output_file = os.path.join(RESULTS_DIR, 'per_datatype_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Per-datatype comparison saved to: {output_file}")
    print(f"  Analyzed {len(all_datatypes)} data types across {len(methods)} methods")

    return comparison


def generate_category_analysis():
    """Generate comprehensive analysis by colormap categories."""
    print("\n" + "="*80)
    print("GENERATING CATEGORY ANALYSIS")
    print("="*80)

    # Get all method configurations dynamically
    methods = get_all_method_configs()

    if not methods:
        print("  ERROR: No method results found!")
        return None

    # Load all per-datatype summaries
    per_datatype_data = {}
    for method in methods:
        data = load_method_per_datatype(method)
        if data:
            per_datatype_data[method] = data

    if not per_datatype_data:
        print("  ERROR: No per-datatype data found")
        return None

    # Collect all datatypes
    all_datatypes = set()
    for method_data in per_datatype_data.values():
        all_datatypes.update(method_data.get('per_datatype_results', {}).keys())

    # Group datatypes by category
    datatypes_by_category = defaultdict(list)
    for datatype in all_datatypes:
        category = get_colormap_category(datatype)
        datatypes_by_category[category].append(datatype)

    # Analyze each category
    category_analysis = {
        'categories': {},
        'overall_insights': {},
        'method_performance_by_category': {}
    }

    # Track individual colormap performance
    colormap_performance = {}

    for category, datatypes in datatypes_by_category.items():
        category_info = COLORMAP_CATEGORIES.get(category, {
            'description': 'Unknown category',
            'characteristics': 'N/A',
            'pros': 'N/A',
            'cons': 'N/A',
            'expected_performance': 'Unknown'
        })

        category_stats = {
            'datatypes': datatypes,
            'count': len(datatypes),
            'description': category_info['description'],
            'characteristics': category_info['characteristics'],
            'pros': category_info['pros'],
            'cons': category_info['cons'],
            'expected_performance': category_info['expected_performance'],
            'methods': {},
            'individual_colormap_performance': {}
        }

        # Track performance for each individual colormap
        for datatype in datatypes:
            colormap_results = {}
            for method in methods:
                if method not in per_datatype_data:
                    continue
                dt_results = per_datatype_data[method].get('per_datatype_results', {}).get(datatype, {})
                if dt_results and dt_results.get('total_pairs', 0) > 0:
                    colormap_results[method] = {
                        'success_rate': dt_results.get('success_rate', 0),
                        'mean_mace': dt_results.get('mace_statistics', {}).get('mean'),
                        'mean_f1': dt_results.get('matching_statistics', {}).get('mean_f1_score')
                    }

            if colormap_results:
                avg_success = np.mean([r['success_rate'] for r in colormap_results.values()])
                category_stats['individual_colormap_performance'][datatype] = {
                    'avg_success_rate_across_methods': float(avg_success),
                    'by_method': colormap_results
                }
                colormap_performance[datatype] = float(avg_success)

        # Analyze each method's performance on this category
        for method in methods:
            if method not in per_datatype_data:
                continue

            method_results = []
            for datatype in datatypes:
                dt_results = per_datatype_data[method].get('per_datatype_results', {}).get(datatype, {})
                if dt_results and dt_results.get('total_pairs', 0) > 0:
                    method_results.append(dt_results)

            if method_results:
                success_rates = [r.get('success_rate', 0) for r in method_results]
                mace_values = [r.get('mace_statistics', {}).get('mean') for r in method_results
                              if r.get('mace_statistics', {}).get('mean') is not None]
                times = [r.get('performance_statistics', {}).get('mean_computation_time') for r in method_results
                        if r.get('performance_statistics', {}).get('mean_computation_time') is not None]
                precisions = [r.get('matching_statistics', {}).get('mean_precision') for r in method_results
                            if r.get('matching_statistics', {}).get('mean_precision') is not None]
                f1_scores = [r.get('matching_statistics', {}).get('mean_f1_score') for r in method_results
                            if r.get('matching_statistics', {}).get('mean_f1_score') is not None]

                category_stats['methods'][method] = {
                    'avg_success_rate': float(np.mean(success_rates)) if success_rates else 0,
                    'avg_mace': float(np.mean(mace_values)) if mace_values else None,
                    'avg_time': float(np.mean(times)) if times else None,
                    'avg_precision': float(np.mean(precisions)) if precisions else None,
                    'avg_f1_score': float(np.mean(f1_scores)) if f1_scores else None,
                    'samples': len(method_results)
                }

        # Find best method for this category
        if category_stats['methods']:
            best_by_success = max(
                category_stats['methods'].items(),
                key=lambda x: x[1]['avg_success_rate']
            )
            best_by_accuracy = min(
                [(m, s) for m, s in category_stats['methods'].items() if s['avg_mace'] is not None],
                key=lambda x: x[1]['avg_mace'],
                default=(None, None)
            )

            category_stats['best_method_by_success'] = best_by_success[0]
            category_stats['best_method_by_accuracy'] = best_by_accuracy[0] if best_by_accuracy[0] else None

            # Calculate category difficulty
            all_success_rates = [s['avg_success_rate'] for s in category_stats['methods'].values()]
            category_stats['difficulty_score'] = 1.0 - float(np.mean(all_success_rates)) if all_success_rates else 1.0
            category_stats['avg_success_rate_across_methods'] = float(np.mean(all_success_rates)) if all_success_rates else 0

            # Performance gap analysis
            success_rates_list = [s['avg_success_rate'] for s in category_stats['methods'].values()]
            category_stats['performance_gap'] = float(max(success_rates_list) - min(success_rates_list)) if success_rates_list else 0

        category_analysis['categories'][category] = category_stats

    # Overall insights
    if category_analysis['categories']:
        categories_by_difficulty = sorted(
            category_analysis['categories'].items(),
            key=lambda x: x[1].get('difficulty_score', 1.0),
            reverse=True
        )

        # Best and worst colormaps
        if colormap_performance:
            best_colormap = max(colormap_performance.items(), key=lambda x: x[1])
            worst_colormap = min(colormap_performance.items(), key=lambda x: x[1])
            performance_range = best_colormap[1] - worst_colormap[1]

            sorted_colormaps = sorted(colormap_performance.items(), key=lambda x: x[1], reverse=True)
            top_5 = sorted_colormaps[:5]
            bottom_5 = sorted_colormaps[-5:]

            category_analysis['overall_insights'] = {
                'easiest_category': categories_by_difficulty[-1][0] if categories_by_difficulty else None,
                'hardest_category': categories_by_difficulty[0][0] if categories_by_difficulty else None,
                'categories_ranked_by_difficulty': [
                    {
                        'category': cat,
                        'difficulty_score': stats.get('difficulty_score', 1.0),
                        'avg_success_rate': stats.get('avg_success_rate_across_methods', 0),
                        'performance_gap': stats.get('performance_gap', 0),
                        'description': stats.get('description', '')
                    }
                    for cat, stats in categories_by_difficulty
                ],
                'best_colormap': {
                    'name': best_colormap[0],
                    'avg_success_rate': best_colormap[1],
                    'category': get_colormap_category(best_colormap[0])
                },
                'worst_colormap': {
                    'name': worst_colormap[0],
                    'avg_success_rate': worst_colormap[1],
                    'category': get_colormap_category(worst_colormap[0])
                },
                'performance_range': {
                    'range': performance_range,
                    'interpretation': f"Performance varies by {performance_range*100:.1f}% between best and worst colormaps"
                },
                'top_5_colormaps': [
                    {'name': name, 'avg_success_rate': rate, 'category': get_colormap_category(name)}
                    for name, rate in top_5
                ],
                'bottom_5_colormaps': [
                    {'name': name, 'avg_success_rate': rate, 'category': get_colormap_category(name)}
                    for name, rate in bottom_5
                ]
            }

        # Method performance across categories
        for method in methods:
            method_category_performance = {}
            for category, stats in category_analysis['categories'].items():
                if method in stats['methods']:
                    method_category_performance[category] = stats['methods'][method]

            if method_category_performance:
                best_category = max(
                    method_category_performance.items(),
                    key=lambda x: x[1]['avg_success_rate']
                )
                worst_category = min(
                    method_category_performance.items(),
                    key=lambda x: x[1]['avg_success_rate']
                )

                category_analysis['method_performance_by_category'][method] = {
                    'best_category': best_category[0],
                    'worst_category': worst_category[0],
                    'performance_by_category': method_category_performance
                }

    # Save analysis
    output_file = os.path.join(RESULTS_DIR, 'category_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(category_analysis, f, indent=2)

    print(f"\n  Category analysis saved to: {output_file}")
    print(f"  Analyzed {len(datatypes_by_category)} categories with {len(all_datatypes)} data types")

    return category_analysis


def generate_ransac_variant_comparison():
    """Generate comparison analysis across RANSAC variants."""
    print("\n" + "="*80)
    print("GENERATING RANSAC VARIANT COMPARISON")
    print("="*80)

    # Get all method configurations
    all_methods = get_all_method_configs()

    # Filter only RANSAC variant results
    ransac_methods = [m for m in all_methods if '_RANSAC' in m or '_MLESAC' in m or '_PROSAC' in m]

    if not ransac_methods:
        print("  No RANSAC variant results found. Skipping RANSAC comparison.")
        return {}

    print(f"  Found {len(ransac_methods)} RANSAC variant configurations")

    # Parse method names
    variant_data = defaultdict(lambda: defaultdict(dict))

    for method_full in ransac_methods:
        parts = method_full.rsplit('_', 1)
        if len(parts) == 2:
            base_method, ransac_variant = parts
            summary = load_method_summary(method_full)
            if summary:
                variant_data[ransac_variant][base_method] = summary

    if not variant_data:
        print("  No valid RANSAC variant data found.")
        return {}

    # Analyze each RANSAC variant
    ransac_analysis = {
        'variants_compared': list(variant_data.keys()),
        'base_methods': list(set([m for v in variant_data.values() for m in v.keys()])),
        'variants': {},
        'rankings': {},
        'best_variant': {}
    }

    for variant, methods in variant_data.items():
        variant_stats = {
            'variant': variant,
            'methods_tested': list(methods.keys()),
            'overall_statistics': {},
            'per_method_performance': {}
        }

        # Aggregate statistics
        all_success_rates = []
        all_mace = []
        all_times = []
        all_precisions = []
        all_f1_scores = []

        for method_name, summary in methods.items():
            all_success_rates.append(summary.get('success_rate', 0))

            mace_stats = summary.get('mace_statistics', {})
            if 'mean' in mace_stats:
                all_mace.append(mace_stats['mean'])

            perf_stats = summary.get('performance_statistics', {})
            if 'mean_computation_time' in perf_stats:
                all_times.append(perf_stats['mean_computation_time'])

            matching_stats = summary.get('matching_statistics', {})
            if 'mean_precision' in matching_stats:
                all_precisions.append(matching_stats['mean_precision'])
            if 'mean_f1_score' in matching_stats:
                all_f1_scores.append(matching_stats['mean_f1_score'])

            variant_stats['per_method_performance'][method_name] = {
                'success_rate': summary.get('success_rate', 0),
                'mean_mace': mace_stats.get('mean', 0),
                'mean_time': perf_stats.get('mean_computation_time', 0),
                'mean_precision': matching_stats.get('mean_precision', 0),
                'mean_f1_score': matching_stats.get('mean_f1_score', 0)
            }

        variant_stats['overall_statistics'] = {
            'avg_success_rate': float(np.mean(all_success_rates)) if all_success_rates else 0,
            'avg_mace': float(np.mean(all_mace)) if all_mace else 0,
            'avg_computation_time': float(np.mean(all_times)) if all_times else 0,
            'avg_precision': float(np.mean(all_precisions)) if all_precisions else 0,
            'avg_f1_score': float(np.mean(all_f1_scores)) if all_f1_scores else 0
        }

        ransac_analysis['variants'][variant] = variant_stats

    # Rankings
    if ransac_analysis['variants']:
        by_success = sorted(
            ransac_analysis['variants'].items(),
            key=lambda x: x[1]['overall_statistics']['avg_success_rate'],
            reverse=True
        )

        ransac_analysis['rankings'] = {
            'by_success_rate': [
                {'variant': v[0], 'avg_success_rate': v[1]['overall_statistics']['avg_success_rate']}
                for v in by_success
            ],
            'by_accuracy': sorted(
                [{'variant': k, 'avg_mace': v['overall_statistics']['avg_mace']}
                 for k, v in ransac_analysis['variants'].items() if v['overall_statistics']['avg_mace'] > 0],
                key=lambda x: x['avg_mace']
            ),
            'by_speed': sorted(
                [{'variant': k, 'avg_time': v['overall_statistics']['avg_computation_time']}
                 for k, v in ransac_analysis['variants'].items() if v['overall_statistics']['avg_computation_time'] > 0],
                key=lambda x: x['avg_time']
            )
        }

        ransac_analysis['best_variant'] = {
            'by_success_rate': by_success[0][0] if by_success else 'N/A',
            'by_accuracy': min(ransac_analysis['variants'].items(),
                             key=lambda x: x[1]['overall_statistics']['avg_mace'] or float('inf'))[0],
            'by_speed': min(ransac_analysis['variants'].items(),
                          key=lambda x: x[1]['overall_statistics']['avg_computation_time'] or float('inf'))[0]
        }

    # Save analysis
    output_file = os.path.join(RESULTS_DIR, 'ransac_variant_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(ransac_analysis, f, indent=2)

    print(f"\n  RANSAC variant comparison saved to: {output_file}")
    print(f"  Compared {len(variant_data)} RANSAC variants")

    return ransac_analysis


def main():
    """Generate all comparison results."""
    print("="*80)
    print("GENERATING COMPARISON RESULTS FOR TRADITIONAL METHODS")
    print("="*80)

    print(f"\nResults directory: {RESULTS_DIR}")

    if not os.path.exists(RESULTS_DIR):
        print(f"\nERROR: Results directory not found: {RESULTS_DIR}")
        print("Please run traditional methods first.")
        return

    # Get all method configurations
    methods = get_all_method_configs()
    print(f"\nFound {len(methods)} method configurations to analyze")

    try:
        cross_method = generate_cross_method_comparison()
        performance_matrix = generate_performance_matrix_csv()
        per_datatype = generate_per_datatype_comparison()
        category_analysis = generate_category_analysis()
        ransac_comparison = generate_ransac_variant_comparison()

        print("\n" + "="*80)
        print("COMPARISON RESULTS GENERATION COMPLETE")
        print("="*80)

        print("\nGenerated files:")
        print(f"  1. cross_method_comparison.json")
        print(f"  2. performance_matrix.csv")
        print(f"  3. per_datatype_comparison.json")
        print(f"  4. category_analysis.json")
        print(f"  5. ransac_variant_comparison.json")

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

        print("\n" + "="*80)

    except Exception as e:
        print(f"\nERROR: Failed to generate comparison results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

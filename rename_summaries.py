"""
rename_summaries.py

Renames summary files to match their parent directory names.
Fixes the naming mismatch between directory names (e.g., SIFT_RANSAC)
and summary file names (e.g., SIFT_summary.json).

Author: Homography Benchmarking Project
Date: 2025
"""

import os
import sys

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'Traditional_methods_results')


def rename_summary_files():
    """
    Rename summary files to match their parent directory names.

    Example:
        Directory: SIFT_RANSAC/
        Old: SIFT_summary.json
        New: SIFT_RANSAC_summary.json
    """
    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return False

    print(f"Scanning results directory: {RESULTS_DIR}\n")

    renamed_count = 0
    error_count = 0

    # Iterate through all directories
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)

        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue

        # Check if this is a method directory (contains underscore for RANSAC variant)
        if '_' not in item:
            continue

        # Extract base method name (e.g., SIFT from SIFT_RANSAC)
        base_method = item.split('_')[0]

        # Define old and new file names
        files_to_rename = [
            (f'{base_method}_summary.json', f'{item}_summary.json'),
            (f'{base_method}_per_datatype_summary.json', f'{item}_per_datatype_summary.json')
        ]

        print(f"Processing directory: {item}/")

        for old_name, new_name in files_to_rename:
            old_path = os.path.join(item_path, old_name)
            new_path = os.path.join(item_path, new_name)

            # Check if old file exists
            if os.path.exists(old_path):
                try:
                    # Check if new file already exists
                    if os.path.exists(new_path):
                        print(f"  [SKIP] {new_name} already exists")
                        continue

                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"  [OK] Renamed: {old_name} -> {new_name}")
                    renamed_count += 1

                except Exception as e:
                    print(f"  [ERROR] renaming {old_name}: {str(e)}")
                    error_count += 1
            else:
                print(f"  [SKIP] {old_name} not found (may already be renamed)")

        print()

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files renamed: {renamed_count}")
    print(f"Errors: {error_count}")

    return error_count == 0


if __name__ == "__main__":
    success = rename_summary_files()
    sys.exit(0 if success else 1)

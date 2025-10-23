"""
Script to update all traditional method classes to support new RANSAC variants and matching parameters.
"""

import re

files_to_update = [
    'traditional_orb.py',
    'traditional_akaze.py',
    'traditional_brisk.py',
    'traditional_kaze.py'
]

for filename in files_to_update:
    print(f"Updating {filename}...")

    with open(filename, 'r') as f:
        content = f.read()

    # Find the super().__init__ call and update it
    # Pattern: super().__init__('METHOD', use_flann, ransac_reproj_threshold)
    pattern = r"super\(\).__init__\('([^']+)',\s*use_flann,\s*ransac_reproj_threshold\)"
    replacement = r"super().__init__('\1', use_flann, ransac_reproj_threshold,\n                        ransac_method, use_advanced_matching, distance_ratio)"

    content = re.sub(pattern, replacement, content)

    # Update __init__ signature
    # Add the new parameters before the closing parenthesis of __init__
    # Find: ransac_reproj_threshold=5.0):
    # Replace with: ransac_reproj_threshold=5.0, ransac_method='RANSAC',
    #               use_advanced_matching=True, distance_ratio=0.75):

    pattern = r"ransac_reproj_threshold=5\.0\):"
    replacement = r"""ransac_reproj_threshold=5.0, ransac_method='RANSAC',
                 use_advanced_matching=True, distance_ratio=0.75):"""

    content = re.sub(pattern, replacement, content)

    # Update the docstring to include new parameters
    # Find the end of the Args section (before ransac_reproj_threshold line)
    pattern = r"(\s+ransac_reproj_threshold:\s+RANSAC reprojection threshold)"
    replacement = r"""\1
            ransac_method: RANSAC variant ('RANSAC', 'MLESAC', 'PROSAC')
            use_advanced_matching: Use advanced FLANN matching with ratio test
            distance_ratio: Lowe's ratio test threshold"""

    content = re.sub(pattern, replacement, content)

    with open(filename, 'w') as f:
        f.write(content)

    print(f"  [OK] Updated {filename}")

print("\nAll files updated successfully!")

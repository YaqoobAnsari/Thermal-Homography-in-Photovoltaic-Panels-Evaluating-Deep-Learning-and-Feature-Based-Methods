"""
Test script to verify RANSAC variants and FLANN matching integration.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.traditional_sift import SIFTHomographyEstimator

def test_ransac_variants():
    """Test that different RANSAC variants can be instantiated."""
    print("Testing RANSAC variants integration...")

    # Test RANSAC (default)
    print("\n1. Testing RANSAC...")
    try:
        estimator_ransac = SIFTHomographyEstimator(
            ransac_method='RANSAC',
            use_advanced_matching=True
        )
        print(f"   [OK] RANSAC estimator created: {estimator_ransac.ransac_method}")
    except Exception as e:
        print(f"   [ERROR] RANSAC: {e}")
        return False

    # Test MLESAC
    print("\n2. Testing MLESAC...")
    try:
        estimator_mlesac = SIFTHomographyEstimator(
            ransac_method='MLESAC',
            use_advanced_matching=True
        )
        print(f"   [OK] MLESAC estimator created: {estimator_mlesac.ransac_method}")
    except Exception as e:
        print(f"   [ERROR] MLESAC: {e}")
        return False

    # Test PROSAC
    print("\n3. Testing PROSAC...")
    try:
        estimator_prosac = SIFTHomographyEstimator(
            ransac_method='PROSAC',
            use_advanced_matching=True
        )
        print(f"   [OK] PROSAC estimator created: {estimator_prosac.ransac_method}")
    except Exception as e:
        print(f"   [ERROR] PROSAC: {e}")
        return False

    # Test legacy matching (backwards compatibility)
    print("\n4. Testing legacy matching (backwards compatibility)...")
    try:
        estimator_legacy = SIFTHomographyEstimator(
            use_advanced_matching=False
        )
        print(f"   [OK] Legacy matcher created")
    except Exception as e:
        print(f"   [ERROR] Legacy: {e}")
        return False

    print("\n" + "="*60)
    print("SUCCESS: All RANSAC variants can be instantiated!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_ransac_variants()
    sys.exit(0 if success else 1)

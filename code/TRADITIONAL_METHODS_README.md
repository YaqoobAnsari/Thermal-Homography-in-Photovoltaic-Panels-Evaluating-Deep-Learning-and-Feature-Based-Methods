# Traditional Methods - Quick Reference Guide

## Overview

Six traditional feature-based homography estimation methods have been implemented:

1. **SIFT** - Scale-Invariant Feature Transform (Best accuracy, slower)
2. **SURF** - Speeded-Up Robust Features (Fast, requires opencv-contrib)
3. **ORB** - Oriented FAST and Rotated BRIEF (Very fast, binary)
4. **AKAZE** - Accelerated-KAZE (Good balance)
5. **BRISK** - Binary Robust Invariant Scalable Keypoints (Fast, binary)
6. **KAZE** - Nonlinear scale space (Accurate, slower)

---

## Quick Start

### Run All Methods (Recommended)
```bash
cd C:\Users\ansar\Desktop\Workstation\CMUQ\Benchmarking
python code/run_all_traditional_methods.py
```

### Run Individual Methods
```bash
# SIFT (recommended first)
python code/traditional_sift.py

# ORB (fastest)
python code/traditional_orb.py

# AKAZE (good balance)
python code/traditional_akaze.py

# BRISK (fast binary)
python code/traditional_brisk.py

# KAZE (accurate)
python code/traditional_kaze.py

# SURF (requires opencv-contrib-python)
python code/traditional_surf.py
```

---

## Prerequisites

### Required
```bash
pip install numpy opencv-python matplotlib scikit-image scipy tifffile
```

### For SURF Only
```bash
pip install opencv-contrib-python
```

**Note**: SURF is patented and requires the non-free opencv-contrib package.

---

## Method Comparison

| Method | Speed | Accuracy | Descriptor | Patent-Free | Best For |
|--------|-------|----------|------------|-------------|----------|
| **SIFT** | Slow | Excellent | Float (128) | Yes (since 2020) | Best accuracy |
| **SURF** | Fast | Very Good | Float (64/128) | No | Speed+accuracy |
| **ORB** | Very Fast | Good | Binary (256) | Yes | Real-time |
| **AKAZE** | Fast | Very Good | Binary/Float | Yes | Balance |
| **BRISK** | Very Fast | Good | Binary (512) | Yes | Real-time |
| **KAZE** | Slow | Excellent | Float (64/128) | Yes | Challenging scenes |

---

## Pipeline Overview

Each method follows the same pipeline:

```
1. Load patch A and patch B
2. Detect keypoints in both patches
3. Compute descriptors for keypoints
4. Match descriptors (FLANN or BFMatcher)
5. Apply Lowe's ratio test (0.75 threshold)
6. Estimate homography using RANSAC
7. Compute corner error vs. ground truth
8. Save results
```

---

## Configuration

### Common Parameters (in base class)

```python
use_flann = True                   # Use FLANN (faster) vs BFMatcher
ransac_reproj_threshold = 5.0     # RANSAC reprojection threshold (pixels)
ratio_threshold = 0.75             # Lowe's ratio test threshold
min_matches = 4                    # Minimum matches for homography
```

### Method-Specific Parameters

#### SIFT
```python
nfeatures = 0                      # 0 = all features
contrastThreshold = 0.04           # Higher = fewer features
edgeThreshold = 10                 # Higher = more edge-like features
```

#### ORB
```python
nfeatures = 1000                   # Max features to detect
scaleFactor = 1.2                  # Pyramid scale factor
nlevels = 8                        # Pyramid levels
```

#### AKAZE
```python
threshold = 0.001                  # Detector response threshold
descriptor_type = MLDB             # Binary (MLDB) or float (KAZE)
```

#### BRISK
```python
thresh = 30                        # AGAST threshold
octaves = 3                        # Scale space octaves
```

#### KAZE
```python
threshold = 0.001                  # Detector threshold
extended = False                   # 64 vs 128 descriptors
```

---

## Output Structure

Results are saved to:
```
results/Traditional_methods_results/
├── SIFT/
│   ├── raw_results.json
│   ├── preprocessed_results.json
│   ├── colormap_viridis_results.json
│   ├── ... (27 colormap results)
│   └── SIFT_summary.json
├── ORB/
├── AKAZE/
├── BRISK/
├── KAZE/
└── SURF/
```

---

## Results Format

### Detailed Results (per data type)
Each `*_results.json` contains per-pair results:

```json
{
  "success": true,
  "image_name": "image_001",
  "pair_name": "pair_0001",
  "mean_corner_error": 2.345,
  "max_corner_error": 4.567,
  "corner_errors": [2.1, 2.3, 2.5, 2.4],
  "num_keypoints_A": 423,
  "num_keypoints_B": 398,
  "num_matches": 156,
  "num_inliers": 142,
  "inlier_ratio": 0.910,
  "computation_time": 0.045,
  "H_estimated": [[...], [...], [...]],
  "H_ground_truth": [[...], [...], [...]]
}
```

### Summary Results
Each `METHOD_summary.json` contains aggregated statistics:

```json
{
  "method": "SIFT",
  "total_pairs": 1450,
  "successful_pairs": 1389,
  "failed_pairs": 61,
  "success_rate": 0.958,
  "mean_corner_error": 3.245,
  "median_corner_error": 2.876,
  "std_corner_error": 1.234,
  "mean_computation_time": 0.052,
  "mean_keypoints": 756.3,
  "mean_matches": 234.5,
  "mean_inliers": 198.7
}
```

---

## Evaluation Metrics

### Mean Corner Error (MCE)
- Average pixel distance between transformed corners
- **Lower is better**
- Good: < 5 pixels, Excellent: < 2 pixels

### Success Rate
- Percentage of pairs with valid homography estimation
- **Higher is better**
- Good: > 80%, Excellent: > 95%

### Inlier Ratio
- Ratio of RANSAC inliers to total matches
- **Higher is better**
- Good: > 0.5, Excellent: > 0.8

### Computation Time
- Time to process one pair (detection + matching + RANSAC)
- **Lower is better**
- Real-time: < 30ms, Fast: < 100ms

---

## Troubleshooting

### SURF Not Available
```
ERROR: SURF is not available.
```
**Solution**: Install opencv-contrib-python
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### No Homography Pairs Found
```
ERROR: No data types found in Homography Pairs directory
```
**Solution**: Run homography pair generation first
```bash
python code/generate_homography_pairs.py
```

### FLANN Matcher Error
```
Matching error: ...
```
**Solution**: Switch to BFMatcher by setting `use_flann=False` in the method constructor

### Insufficient Keypoints
If many pairs fail with "No keypoints detected":
- Lower detection thresholds (e.g., SIFT `contrastThreshold = 0.02`)
- Increase max features (e.g., ORB `nfeatures = 2000`)
- Try different methods (SIFT typically detects more features)

---

## Performance Tips

### For Speed
1. Use ORB or BRISK
2. Enable FLANN matcher
3. Reduce max features (ORB: `nfeatures=500`)
4. Use binary descriptors (ORB, BRISK, AKAZE-MLDB)

### For Accuracy
1. Use SIFT or KAZE
2. Increase features (SIFT: `nfeatures=0` for unlimited)
3. Lower RANSAC threshold (e.g., 3.0 pixels)
4. Use float descriptors (SIFT, KAZE, AKAZE-KAZE)

### For Balance
1. Use AKAZE
2. Keep default parameters
3. Use FLANN matcher

---

## Customization

To modify parameters, edit the method's main() function:

```python
# Example: More aggressive SIFT
estimator = SIFTHomographyEstimator(
    nfeatures=0,              # Unlimited features
    contrastThreshold=0.02,   # Lower threshold = more features
    edgeThreshold=20,         # Higher = more edge features
    ransac_reproj_threshold=3.0  # Stricter RANSAC
)
```

---

## Expected Runtime

For full dataset (5 images × 29 data types × 10 pairs = 1,450 pairs):

| Method | Estimated Time |
|--------|----------------|
| SIFT   | ~10-15 minutes |
| ORB    | ~3-5 minutes   |
| AKAZE  | ~5-8 minutes   |
| BRISK  | ~3-5 minutes   |
| KAZE   | ~12-18 minutes |
| SURF   | ~8-12 minutes  |

**Total (all methods)**: ~40-60 minutes

---

## Next Steps

After running traditional methods:

1. **Analyze Results**
   ```bash
   # Results are in JSON format
   # Use Python, Pandas, or custom scripts to analyze
   ```

2. **Compare Methods**
   - Compare summary.json files across methods
   - Identify best method for each colormap
   - Analyze failure cases

3. **Deep Learning Methods**
   - Proceed to DL-based homography estimation
   - Compare traditional vs. DL performance

---

## Contact

For issues or questions about traditional methods implementation, refer to the main project README or contact the project team.

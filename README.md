# Homography Benchmarking Project

A comprehensive benchmarking framework for evaluating homography estimation methods on thermal imagery. This project compares traditional computer vision approaches across **3 dimensions**:
- **5 Feature Detection Methods**: SIFT, ORB, AKAZE, BRISK, KAZE
- **3 RANSAC Variants**: RANSAC, MLESAC, PROSAC
- **29 Colormap Variations**: Raw, preprocessed, and 27 colormaps across 8 scientific categories

**Total Benchmarking Capacity**: 15 method×RANSAC configurations × 1,642 homography pairs = **24,630 evaluations** with comprehensive multi-dimensional analysis.

---

## 🚀 Quick Start

**Want to run everything?** Just execute:
```bash
python code/run_all_variants.py
```

This single command will:
✅ Run all 5 methods (SIFT, ORB, AKAZE, BRISK, KAZE)
✅ Test all 3 RANSAC variants (RANSAC, MLESAC, PROSAC)
✅ Evaluate across all 29 colormaps
✅ Generate 5 comprehensive comparison files
✅ Provide rich insights across methods, RANSAC variants, and colormap categories

**Time**: ~90-150 minutes | **Output**: 15 result directories + 5 analysis files

**Need faster results?** Use `python code/run_all_traditional_methods.py` (~30-45 min, 5 configs only)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Organization](#dataset-organization)
- [Workflow](#workflow)
- [Scripts](#scripts)
- [Getting Started](#getting-started)
- [Traditional Methods](#traditional-methods)
- [Deep Learning Methods](#deep-learning-methods)
- [Results](#results)
- [Requirements](#requirements)

---

## Overview

### Goal
Benchmark and compare homography estimation performance across:
- **Traditional Methods**: Feature-based approaches (SIFT, ORB, AKAZE, etc.)
- **Deep Learning Methods**: Neural network-based approaches (HomographyNet, etc.)

### Dataset
- **39 thermal TIFF images** (14-bit, 512×640 resolution)
- **29 data variants** per image:
  - 1 raw thermal image
  - 1 preprocessed thermal image
  - 27 colormap-applied variants

### Homography Pairs
- **42 pairs per image** across all 29 variants
- **Total: 1,642 homography pairs** for comprehensive evaluation

---

## Project Structure

```
Benchmarking/
├── README.md                          # This file
├── benchmarking_dataset/              # Main dataset directory
│   ├── Image Data/                    # All image data
│   │   ├── raw/                       # Original 39 TIFF images (14-bit)
│   │   ├── preprocessed/              # Preprocessed images (14-bit preserved)
│   │   └── colormap_outputs/          # 27 colormap variants
│   │       ├── viridis/
│   │       ├── plasma/
│   │       ├── inferno/
│   │       ├── magma/
│   │       ├── cividis/
│   │       ├── hot/
│   │       ├── cool/
│   │       ├── copper/
│   │       ├── Greys/
│   │       ├── Blues/
│   │       ├── Oranges/
│   │       ├── Reds/
│   │       ├── coolwarm/
│   │       ├── seismic/
│   │       ├── RdBu/
│   │       ├── BrBG/
│   │       ├── PiYG/
│   │       ├── twilight/
│   │       ├── twilight_shifted/
│   │       ├── hsv/
│   │       ├── tab10/
│   │       ├── Set1/
│   │       ├── Set2/
│   │       ├── Pastel1/
│   │       ├── terrain/
│   │       ├── nipy_spectral/
│   │       └── flag/
│   └── Homography Pairs/              # Generated homography training pairs
│       ├── raw/                       # Pairs from raw images
│       ├── preprocessed/              # Pairs from preprocessed images
│       └── colormap_*/                # Pairs from each colormap variant
├── code/                              # Main scripts only
│   ├── preprocess_and_colormap.py    # Preprocessing & colormap application
│   ├── generate_homography_pairs.py  # Homography pair generation
│   ├── run_all_traditional_methods.py # Quick benchmark (5 methods, default RANSAC)
│   └── run_all_variants.py           # Comprehensive benchmark (15 method×RANSAC combos)
├── Utils/                             # Utility modules (NEW - organized code)
│   ├── __init__.py                   # Package initialization
│   ├── traditional_base.py           # Base class for traditional methods
│   ├── traditional_sift.py           # SIFT implementation
│   ├── traditional_orb.py            # ORB implementation
│   ├── traditional_akaze.py          # AKAZE implementation
│   ├── traditional_brisk.py          # BRISK implementation
│   ├── traditional_kaze.py           # KAZE implementation
│   ├── enhanced_evaluation.py        # Comprehensive evaluation metrics
│   ├── ransac_variants.py            # RANSAC/MLESAC/PROSAC implementations
│   ├── generate_comparison_results.py # Cross-method and category analysis
│   └── regenerate_summaries.py       # Summary regeneration utility
├── results/                           # Benchmarking results
│   └── Traditional_methods_results/  # Results from traditional approaches
│       ├── SIFT/ ORB/ AKAZE/ BRISK/ KAZE/ # Per-method results (5 directories)
│       │   OR (when using run_all_variants.py):
│       ├── SIFT_RANSAC/ SIFT_MLESAC/ SIFT_PROSAC/ # 15 method×variant combos
│       ├── ORB_RANSAC/ ORB_MLESAC/ ORB_PROSAC/
│       ├── AKAZE_RANSAC/ AKAZE_MLESAC/ AKAZE_PROSAC/
│       ├── BRISK_RANSAC/ BRISK_MLESAC/ BRISK_PROSAC/
│       ├── KAZE_RANSAC/ KAZE_MLESAC/ KAZE_PROSAC/
│       ├── cross_method_comparison.json    # Cross-method rankings
│       ├── performance_matrix.csv          # 20 metrics × all configs
│       ├── per_datatype_comparison.json    # 29 datatypes compared
│       ├── category_analysis.json          # Colormap category insights
│       └── ransac_variant_comparison.json  # RANSAC variant analysis (NEW)
└── old stuff/                         # Archived/legacy code

```

---

## Dataset Organization

### Image Data Structure

Each data type follows this organization:

```
benchmarking_dataset/Image Data/
└── [data_type]/
    ├── image_001.tiff
    ├── image_002.tiff
    └── ...
```

### Homography Pairs Structure

Each image generates 10 homography pairs:

```
benchmarking_dataset/Homography Pairs/
└── [data_type]/
    └── [image_name]/
        ├── pair_0001/
        │   ├── patch_A.png           # Source patch
        │   ├── patch_B.png           # Target patch (warped)
        │   ├── homography_H.txt      # 3×3 homography matrix
        │   ├── delta_4pt.txt         # 4-point offset (8 values)
        │   └── metadata.txt          # Comprehensive metadata
        ├── pair_0002/
        └── ...
```

### Metadata Format

Each homography pair includes detailed metadata:

```
Data Type: raw
Image Path: /path/to/image.tiff
Image Name: image_001.tiff
Patch Size: (256, 256)
Overlap: 0.4523
Rho (Perturbation): 32

Patch 1 Corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
Patch 2 Corners (Original): [(x1, y1), ...]
Patch 2 Corners (Perturbed): [(x1, y1), ...]

Homography Verification:
  Max Error: 0.8423 pixels
  Mean Error: 0.4521 pixels
  Valid: True
```

---

## Workflow

### Phase 1: Data Preparation
1. **Raw Data Collection**: 39 thermal TIFF images (14-bit)
2. **Preprocessing**: Apply noise reduction, glare suppression, anomaly correction
3. **Colormap Application**: Generate 27 colormap variants for visualization

### Phase 2: Homography Pair Generation
1. **Random Patch Selection**: Select patch pairs with controlled overlap
2. **Geometric Perturbation**: Apply corner perturbations (15-50 pixels)
3. **Homography Computation**: Calculate 3×3 transformation matrix
4. **Validation**: Verify homography accuracy (< 2px error tolerance)

### Phase 3: Benchmarking
1. **Traditional Methods**: Run feature-based homography estimation
2. **Deep Learning Methods**: Run neural network-based estimation
3. **Evaluation**: Compare accuracy, robustness, and computational cost

---

## Scripts

### Data Preprocessing & Generation

#### `preprocess_and_colormap.py`
Preprocesses thermal images and generates colormap variants.

**Features:**
- Adaptive contextual anomaly correction
- Glare detection and suppression
- 14-bit data integrity preservation
- 27 colormap variants with CLAHE enhancement

**Usage:**
```bash
python code/preprocess_and_colormap.py
```

**Configuration:**
- `N_IMAGES = 5`: Number of random images to process
- Modify for full dataset processing

---

#### `generate_homography_pairs.py`
Generates synthetic homography training pairs for all data types.

**Features:**
- Processes all 29 data types automatically
- **Proper 14-bit thermal image handling**: Histogram stretching + CLAHE before conversion
- Generates variable number of pairs per image (average 42 pairs/image)
- Validates homography accuracy
- Saves comprehensive metadata

**Usage:**
```bash
python code/generate_homography_pairs.py
```

**Configuration:**
```python
NUM_SAMPLES_PER_IMAGE = 50  # Attempts per image
PATCH_SIZE = (256, 256)     # Patch dimensions
overlap_range = (0.25, 0.55) # Overlap fraction
rho_range = (15, 50)        # Perturbation range (pixels)
```

**14-bit Thermal Image Processing:**
- Percentile-based histogram stretching (2nd-98th percentile)
- CLAHE applied on properly stretched 8-bit data (clipLimit=3.0, tileGridSize=(8,8))
- Ensures proper contrast for feature detection

**Expected Output:**
- 39 images × ~42 pairs × 29 data types = **1,642 homography pairs**

---

### Traditional Methods

All traditional method implementations inherit from `TraditionalHomographyEstimator` base class.

**Implemented Methods:**
- `traditional_sift.py` - SIFT-based homography estimation ✅
- `traditional_orb.py` - ORB-based homography estimation ✅
- `traditional_akaze.py` - AKAZE-based homography estimation ✅
- `traditional_brisk.py` - BRISK-based homography estimation ✅
- `traditional_kaze.py` - KAZE-based homography estimation ✅
- `run_all_traditional_methods.py` - Master script to run all methods ✅

**Usage:**
```bash
# Option 1: Run all methods with default RANSAC (fastest - 5 configurations)
python code/run_all_traditional_methods.py

# Option 2: Run comprehensive benchmark - ALL methods × ALL RANSAC variants (15 configurations)
python code/run_all_variants.py

# Option 3: Run individual method (for testing/debugging)
python code/traditional_sift.py
```

**What Each Script Does:**

1. **`run_all_traditional_methods.py`** (Quick Benchmark - ~30-45 min)
   - Runs 5 methods: SIFT, ORB, AKAZE, BRISK, KAZE
   - Uses default RANSAC with advanced FLANN matching
   - Generates all comparison analyses
   - **Output**: 5 result directories + 4 comparison files

2. **`run_all_variants.py`** (Comprehensive Benchmark - ~90-150 min)
   - Runs 5 methods × 3 RANSAC variants = **15 configurations**
   - Method naming: `METHOD_RANSACVARIANT` (e.g., `SIFT_MLESAC`, `ORB_PROSAC`)
   - Generates all comparisons PLUS RANSAC variant analysis
   - **Output**: 15 result directories + 5 comparison files (includes `ransac_variant_comparison.json`)
   - **Use this for**: Complete performance characterization across methods, RANSAC variants, and colormaps

**Enhanced Evaluation Framework** (`enhanced_evaluation.py`):
Comprehensive evaluation module implementing:
- **MACE (Mean Average Corner Error)**: 4-corner point evaluation
- **Reprojection Error**: 9-point evaluation (corners + edge midpoints + center)
- **RMSE (Root Mean Squared Error)**: Overall transformation accuracy
- **Matching Precision/Recall**: Feature matching quality metrics
- **F1 Score**: Harmonic mean of precision and recall
- **Matrix Error (Frobenius Norm)**: Direct homography matrix comparison
- **Performance Metrics**: Computation time and memory usage
- **Failure Mode Analysis**: Categorized failure types
- **Robustness Metrics**: Error variance, consistency, outlier detection

---

### Deep Learning Methods

**Planned Implementations:**
- Deep learning-based homography estimation models
- Neural network training pipelines
- Model evaluation scripts

---

## Getting Started

### 1. Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Preprocess images and generate colormaps
python code/preprocess_and_colormap.py

# Generate homography pairs
python code/generate_homography_pairs.py
```

### 3. Run Benchmarks

```bash
# Traditional methods (coming soon)
python code/traditional_sift.py

# Deep learning methods (coming soon)
python code/dl_homography_net.py
```

---

## Traditional Methods

Traditional computer vision approaches for homography estimation:

### Feature-Based Methods
- **SIFT (Scale-Invariant Feature Transform)**: Robust to scale and rotation
- **ORB (Oriented FAST and Rotated BRIEF)**: Fast, free alternative to SIFT
- **AKAZE (Accelerated-KAZE)**: Good balance of speed and accuracy
- **SURF (Speeded-Up Robust Features)**: Fast feature detection

### Pipeline
1. **Feature Detection**: Detect keypoints in both patches
2. **Feature Matching**: Match descriptors using FLANN or BFMatcher
3. **Outlier Removal**: RANSAC-based filtering
4. **Homography Estimation**: Compute transformation matrix
5. **Validation**: Evaluate against ground truth

---

## Deep Learning Methods

Neural network-based approaches for direct homography estimation:

### Architectures
- **HomographyNet**: End-to-end regression network
- **Unsupervised approaches**: Self-supervised learning methods
- **Hybrid methods**: Combining traditional and DL techniques

---

## Results

### Storage Structure

Results are organized in a hierarchical structure for efficient access:

```
results/Traditional_methods_results/
├── SIFT/
│   ├── detailed_results/              # Per-datatype detailed results (~15MB)
│   │   ├── raw_results.json
│   │   ├── preprocessed_results.json
│   │   └── colormap_*_results.json (x27)
│   ├── SIFT_summary.json              # Overall statistics (~10KB)
│   └── SIFT_per_datatype_summary.json # Per-datatype stats (~20KB)
├── ORB/ AKAZE/ BRISK/ KAZE/ SURF/    # Same structure for each method
└── (Total: ~100MB for all methods)
```

**Storage Levels:**
1. **Detailed Results**: Individual pair results (for debugging/deep analysis)
2. **Overall Summary**: Aggregated statistics across all data types
3. **Per-Datatype Summary**: Statistics for each of 29 data types

### Evaluation Metrics

#### Core Accuracy Metrics
- **MACE (Mean Average Corner Error)**: Average Euclidean distance across 4 corners
  - Per-corner error breakdown
  - Min/max/std corner errors
- **Reprojection Error**: Error across 9 evaluation points
  - Mean, median, max, std reprojection errors
  - Corners + edge midpoints + center point evaluation
- **RMSE (Root Mean Squared Error)**: Overall transformation accuracy

#### Matching Quality Metrics
- **Matching Precision**: Inliers / Total matches
- **Matching Recall**: Matches / Min keypoints
- **F1 Score**: Harmonic mean of precision and recall
- **Inlier Ratio**: RANSAC inliers / matches

#### Homography Matrix Metrics
- **Frobenius Norm**: Direct matrix distance
- **Max Element Error**: Largest per-element difference
- **Mean Element Error**: Average element-wise difference

#### Performance Metrics
- **Computation Time**: Per-pair processing time
- **Memory Usage**: RAM consumption via psutil

#### Robustness Metrics
- **Success Rate**: Percentage of successful estimations
- **Error Variance**: Spread of error values
- **Consistency Score**: Success rate across all pairs
- **Outlier Ratio**: High-error estimations (>10px)
- **Failure Mode Distribution**: Categorized failure types
  - `insufficient_keypoints`
  - `insufficient_matches`
  - `homography_estimation_failed`
  - `high_error_estimation`
  - `success`

#### Statistical Aggregates
- **Percentiles**: 25th, 75th, 95th error percentiles
- **Mean/Median/Std**: Central tendency and spread

---

## Requirements

### Core Dependencies
```
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-image>=0.18.0
scipy>=1.7.0
tifffile>=2021.7.2
psutil>=5.8.0        # For memory usage tracking
```

### Optional (for Deep Learning)
```
torch>=1.9.0
torchvision>=0.10.0
tensorboard>=2.6.0
```

### Installation
```bash
pip install numpy opencv-python matplotlib scikit-image scipy tifffile psutil
```

---

## Image Preprocessing Details

### Preprocessing Pipeline

1. **Adaptive Contextual Anomaly Correction**
   - Local outlier detection using block-wise statistics
   - Contextual replacement with neighborhood mean
   - Gaussian smoothing (σ=1) for noise reduction

2. **Glare Detection and Suppression**
   - Threshold-based detection (14,000 for 14-bit)
   - Region-based suppression (30% intensity reduction)
   - Area filtering (> 50 pixels)

3. **Contrast Enhancement**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Clip limit: 0.03
   - Preserves local contrast

### Data Integrity
- **14-bit Preservation**: All operations maintain 0-16383 range
- **No Clipping**: Float64 precision during processing
- **Lossless Conversion**: uint16 output format

---

## Homography Generation Details

### Algorithm

1. **Patch 1 Generation**
   - Random position with center bias (Gaussian distribution)
   - Size: 256×256 pixels
   - Boundary validation

2. **Patch 2 Generation**
   - Slide from Patch 1 with controlled overlap (25-55%)
   - 8 directional sliding: horizontal, vertical, diagonal
   - Automatic boundary handling

3. **Corner Perturbation**
   - Random offsets: ±15 to ±50 pixels per corner
   - Maintains image boundary constraints
   - Creates realistic geometric deformation

4. **Homography Computation**
   - 4-point correspondence: Patch 1 corners → Patch 2 perturbed corners
   - OpenCV `getPerspectiveTransform` (DLT algorithm)
   - Returns 3×3 transformation matrix

5. **Validation**
   - Forward projection verification
   - Maximum error threshold: 2.0 pixels
   - Rejects invalid transformations

---

## Recent Improvements & Fixes

### October 2025 - RANSAC Variants & Advanced Matching Enhancements

**RANSAC Variants Implementation**
- **New Module**: `Utils/ransac_variants.py` implementing multiple RANSAC algorithms:
  - **RANSAC** (Random Sample Consensus) - Standard OpenCV implementation
  - **MLESAC** (Maximum Likelihood Estimation SAC) - Probabilistic scoring approach
    - Uses log-likelihood instead of inlier counting
    - Better handling of noise in correspondences
    - Configurable inlier/outlier probability distributions
  - **PROSAC** (Progressive Sample Consensus) - Quality-guided sampling
    - Exploits match quality rankings for faster convergence
    - Progressively expands sampling subset based on match confidence
    - Particularly effective with high-quality feature detectors

**Advanced Feature Matching**
- **FLANN-Based Matcher**: Optimized matching with configurable parameters
  - **KD-Tree** for float descriptors (SIFT, KAZE) - 5 trees, 50 checks
  - **LSH** (Locality-Sensitive Hashing) for binary descriptors (ORB, BRISK, AKAZE)
  - **Lowe's Ratio Test**: Distance ratio threshold (default 0.75) for robust filtering
- **Matching Statistics**: Comprehensive tracking of matching performance
  - Total matches, good matches after ratio test
  - Ratio test pass/fail counts
  - Mean and median distance ratios
  - Matching time per pair

**Integration & Configuration**
- All traditional methods now support RANSAC variant selection via constructor:
  ```python
  # Example: Use MLESAC with FLANN matching
  estimator = SIFTHomographyEstimator(
      ransac_method='MLESAC',          # 'RANSAC', 'MLESAC', or 'PROSAC'
      use_advanced_matching=True,      # Enable FLANN with ratio test
      distance_ratio=0.75              # Lowe's ratio threshold
  )
  ```
- **Backwards Compatible**: Default behavior unchanged (RANSAC with standard matching)
- **Configurable Parameters**:
  - `ransac_reproj_threshold`: Inlier threshold (default 5.0 pixels)
  - `ransac_method`: Algorithm selection
  - `use_advanced_matching`: Enable/disable advanced FLANN matcher
  - `distance_ratio`: Ratio test threshold for match filtering

**Code Reorganization**
- **Created Utils/ Package**: Clean separation of concerns
  - Moved all traditional method implementations
  - Moved enhanced evaluation module
  - Moved comparison and summary generation scripts
  - Added proper `__init__.py` with exports
- **Simplified code/ Directory**: Now contains only 3 main scripts
  - `preprocess_and_colormap.py` - Data preparation
  - `generate_homography_pairs.py` - Pair generation
  - `run_all_traditional_methods.py` - Master benchmarking script
- **Auto-Generation**: `run_all_traditional_methods.py` now automatically:
  - Runs all 5 traditional methods
  - Generates cross-method comparison
  - Generates performance matrix (CSV)
  - Generates per-datatype comparison
  - Generates category analysis with insights

**Comprehensive Comparison Framework**
- **Cross-Method Comparison** (`cross_method_comparison.json`):
  - Rankings by success rate, accuracy (MACE), speed, precision, F1 score
  - Per-method detailed statistics (all enhanced metrics)
  - Overall "best method" analysis across multiple dimensions

- **Performance Matrix** (`performance_matrix.csv`):
  - 20 metrics × 5 methods comparison table
  - Includes: success rate, MACE, reprojection error, RMSE, precision, recall, F1 score,
    computation time, inlier ratio, Frobenius norm, error variance, consistency score, etc.

- **Per-Datatype Comparison** (`per_datatype_comparison.json`):
  - All 29 datatypes (raw, preprocessed, 27 colormaps) analyzed
  - Per-method performance on each datatype
  - Best/worst performing methods per datatype

- **Category Analysis** (`category_analysis.json`):
  - **8 Scientifically Justified Categories**:
    1. **Grayscale** (preprocessed, Greys, raw) - Best for feature preservation
    2. **Perceptual Uniform** (viridis, plasma, inferno, magma, cividis) - Worst performers
    3. **Sequential Single-Hue** (Blues, Reds, Oranges, copper)
    4. **Sequential Multi-Hue** (hot, cool, terrain, nipy_spectral)
    5. **Diverging** (coolwarm, seismic, RdBu, BrBG, PiYG)
    6. **Cyclic** (twilight, twilight_shifted, hsv)
    7. **Qualitative** (Set1, Set2, Pastel1, tab10)
    8. **Miscellaneous** (flag)
  - **Each Category Includes**:
    - Description and characteristics
    - Pros and cons for computer vision
    - Expected vs. actual performance
    - Individual colormap performance tracking
    - Success rate statistics and rankings
  - **Key Findings**:
    - **Best Colormap**: flag (100% success)
    - **Worst Colormap**: cool (0% success)
    - **Hardest Category**: Perceptual Uniform (4.37% avg) - Smooths gradients too much
    - **Easiest Category**: Miscellaneous/Diverging (60%+) - Preserves sharp transitions
    - **Insight**: Human-friendly colormaps (viridis, plasma) worst for CV algorithms

- **RANSAC Variant Comparison** (`ransac_variant_comparison.json`) - **NEW**:
  - **Multi-Dimensional Analysis**:
    - Overall statistics per RANSAC variant (avg success rate, MACE, time, precision, F1)
    - Per-method performance for each RANSAC variant
    - Rankings by success rate, accuracy, and speed
    - Best variant identification across different metrics
  - **Insights Provided**:
    - Which RANSAC variant performs best overall?
    - Which variant is fastest/most accurate?
    - How does each base method (SIFT, ORB, etc.) perform with each variant?
    - Speed vs. accuracy trade-offs across variants
  - **Use Case**: Understand if MLESAC/PROSAC provide advantages over standard RANSAC

**Complete Benchmarking Workflow**:
```bash
# Step 1: Generate homography pairs (one-time setup)
python code/generate_homography_pairs.py

# Step 2: Run comprehensive benchmark (ALL combinations)
python code/run_all_variants.py

# Result: 15 configurations tested, 5 comparison files generated
# - 15 result directories: METHOD_RANSACVARIANT
# - cross_method_comparison.json (15 configs ranked)
# - performance_matrix.csv (20 metrics × 15 configs)
# - per_datatype_comparison.json (29 datatypes × 15 configs)
# - category_analysis.json (8 categories analyzed)
# - ransac_variant_comparison.json (3 variants compared)
```

### October 2025 - Thermal Image Processing & Enhanced Evaluation

**Critical Fix: 14-bit Thermal Image Handling**
- **Issue**: Raw/preprocessed thermal images had 0% success rate due to improper normalization
  - Original: Dividing by `img.max()` made 700-800 values → 235-255 (saturated)
  - Result: Only 21 unique pixel values, no detectable features
- **Solution**: Implemented proper thermal image processing pipeline:
  1. Percentile-based histogram stretching (2nd-98th percentile) on 14-bit data
  2. Convert to 8-bit by dividing by 256 (not by max)
  3. Apply CLAHE (clipLimit=3.0, tileGridSize=8x8) for local contrast
- **Result**: Patch values now span 27-118 with 92 unique values (proper contrast)

**Enhanced Evaluation Framework**
- Created comprehensive `enhanced_evaluation.py` module with 15+ metrics
- Integrated into all traditional methods via `traditional_base.py`
- All results now include:
  - MACE, Reprojection Error, RMSE
  - Matching Precision/Recall/F1 Score
  - Frobenius Norm matrix error
  - Per-pair memory usage tracking
  - Categorized failure mode analysis
  - Robustness metrics (variance, outlier detection)
  - **NEW**: RANSAC variant statistics (method, iterations, estimation time)
  - **NEW**: Matching statistics (ratio test performance, distance ratios)

**Dataset Regeneration**
- Deleted and regenerated all preprocessed images with proper 14-bit handling
- Deleted and regenerated all 1,642 homography pairs with fixed thermal processing
- Re-ran all traditional methods (SIFT, ORB, AKAZE, BRISK, KAZE) with enhanced evaluation

**Performance Improvements**
- Success rates dramatically improved for raw/preprocessed thermal images
- Feature detection now works properly on 14-bit thermal data
- Comprehensive metrics enable deeper analysis of failure modes
- **NEW**: RANSAC variants provide alternative estimation strategies
- **NEW**: Advanced FLANN matching improves match quality and reduces outliers

---

## Known Issues & Limitations

1. **Windows Path Handling**: Ensure paths use proper escaping
2. **Memory Usage**: Processing all 29 variants requires ~8GB RAM
3. **TIFF Support**: Requires `tifffile` library for 14-bit images
4. **Thermal Image Specifics**: Ensure histogram stretching is applied before CLAHE for thermal data

---

## Future Work

- [x] Implement all traditional methods (SIFT, ORB, AKAZE, BRISK, KAZE)
- [x] Create comprehensive evaluation framework
- [x] Fix 14-bit thermal image processing
- [x] Implement RANSAC variants (MLESAC, PROSAC)
- [x] Implement advanced FLANN matching with ratio test
- [x] Code reorganization (Utils/ package structure)
- [x] Comprehensive comparison framework (cross-method, per-datatype, category analysis)
- [x] Scientifically categorize and analyze all 29 colormaps
- [ ] Benchmark RANSAC variants vs standard RANSAC (comparative study)
- [ ] Experiment with different distance ratio thresholds (0.6, 0.7, 0.75, 0.8)
- [ ] Train deep learning models (HomographyNet, unsupervised approaches)
- [ ] Compare traditional vs. deep learning performance
- [ ] Add real-world test cases
- [ ] Expand to larger thermal datasets
- [ ] Investigate domain adaptation techniques
- [ ] Optimize methods for thermal-specific characteristics
- [ ] Explore hybrid traditional-DL approaches

---

## Contributing

This is a research project. For questions or contributions, please contact the project team.

---

## License

Research use only.

---

## Authors

CMUQ Homography Benchmarking Team

---

## Acknowledgments

- Original preprocessing and colormap logic
- Corrected homography generation algorithm
- OpenCV and scikit-image communities

---

## Project Status

**Current Phase**: Traditional methods fully benchmarked with advanced RANSAC variants and comprehensive analysis

| Task | Status |
|------|--------|
| Raw data collection | ✅ Complete (39 images) |
| Preprocessing pipeline | ✅ Complete |
| Colormap generation | ✅ Complete (27 variants) |
| Homography pair generation | ✅ Complete (1,642 pairs) |
| 14-bit thermal image fix | ✅ Complete |
| Enhanced evaluation framework | ✅ Complete (15+ metrics) |
| Traditional methods | ✅ Complete (5 methods) |
| RANSAC variants implementation | ✅ Complete (RANSAC, MLESAC, PROSAC) |
| Advanced FLANN matching | ✅ Complete (with ratio test) |
| Code reorganization | ✅ Complete (Utils/ package) |
| Cross-method comparison | ✅ Complete (rankings, performance matrix) |
| Category analysis | ✅ Complete (8 categories, 29 colormaps) |
| Traditional results compilation | ✅ Complete |
| Deep learning methods | ⏳ Planned |
| RANSAC variant benchmarking | ⏳ Planned (comparative study) |

---

## Quick Reference

### Directory Shortcuts
- **Raw Images**: `benchmarking_dataset/Image Data/raw/`
- **Preprocessed**: `benchmarking_dataset/Image Data/preprocessed/`
- **Homography Pairs**: `benchmarking_dataset/Homography Pairs/`
- **Code**: `code/`
- **Results**: `results/`

### Key Scripts
- **Preprocessing**: `code/preprocess_and_colormap.py`
- **Pair Generation**: `code/generate_homography_pairs.py`
- **Traditional Methods**: `code/traditional_*.py`

### Important Files
- **Homography Matrix**: `homography_H.txt` (3×3 matrix)
- **4-Point Offset**: `delta_4pt.txt` (8 values: Δx₁, Δy₁, ..., Δx₄, Δy₄)
- **Metadata**: `metadata.txt` (comprehensive information)

---

**Last Updated**: October 23, 2025

---

## Key Accomplishments

✅ **Complete Traditional Methods Pipeline**
- 5 methods implemented and benchmarked (SIFT, ORB, AKAZE, BRISK, KAZE)
- 1,642 homography pairs evaluated per method
- 15+ comprehensive evaluation metrics
- Results organized hierarchically (detailed, summary, per-datatype)

✅ **Advanced RANSAC & Matching Enhancements**
- 3 RANSAC variants: RANSAC, MLESAC (probabilistic), PROSAC (quality-guided)
- FLANN-based matching with KD-Tree (float) and LSH (binary) indexing
- Lowe's ratio test for robust match filtering
- Comprehensive matching and RANSAC statistics tracking
- All methods support configurable RANSAC variants

✅ **Comprehensive Analysis Framework**
- **Cross-Method Comparison**: Rankings by success rate, accuracy, speed, precision, F1 score
- **Performance Matrix**: 20 metrics × 5 methods in CSV format
- **Per-Datatype Comparison**: Performance across all 29 datatypes
- **Category Analysis**: 8 scientifically justified colormap categories
  - Best colormap: flag (100%), Worst: cool (0%)
  - Key insight: Perceptual uniform colormaps worst for CV (4.37% avg)

✅ **Thermal Image Processing Fixed**
- Proper 14-bit data handling with histogram stretching
- CLAHE applied correctly for contrast enhancement
- Success rates dramatically improved

✅ **Clean Code Organization**
- **Utils/ Package**: All utility modules properly organized
- **code/ Directory**: Only 3 main scripts (clean entry points)
- **Auto-Generation**: Master script generates all comparisons automatically
- Modular architecture with base class inheritance
- Comprehensive error handling and documentation

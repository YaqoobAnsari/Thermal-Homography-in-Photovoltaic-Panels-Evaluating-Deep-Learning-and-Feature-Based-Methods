import os
import random
import numpy as np
from matplotlib import cm
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter
from skimage import exposure, measure

# Get the project root directory (parent of 'code' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Configuration
N_IMAGES = 5  # Number of random images to process

# Use relative paths from project root
RAW_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Image Data', 'raw')
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Image Data', 'preprocessed')
COLORMAP_BASE_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Image Data', 'colormap_outputs')

# Comprehensive list of colormaps (27 colormaps)
COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'copper', 'Greys', 'Blues',
    'Oranges', 'Reds', 'coolwarm', 'seismic', 'RdBu', 'BrBG', 'PiYG', 'twilight', 'twilight_shifted',
    'hsv', 'tab10', 'Set1', 'Set2', 'Pastel1', 'terrain', 'nipy_spectral', 'flag'
]

# ===========================
# PREPROCESSING FUNCTIONS
# ===========================

def adaptive_local_outlier_detection(image, block_size=50, threshold_factor=2.5):
    """
    Detect outliers locally within blocks of the image.
    Preserves original data type and precision.
    """
    outliers_mask = np.zeros_like(image, dtype=bool)
    height, width = image.shape

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            local_mean = np.mean(block, dtype=np.float64)
            local_std_dev = np.std(block, dtype=np.float64)
            high = local_mean + threshold_factor * local_std_dev
            low = local_mean - threshold_factor * local_std_dev
            outliers_mask[i:i+block_size, j:j+block_size] = (block > high) | (block < low)

    return outliers_mask


def adaptive_contextual_anomaly_correction(image, block_size=50, threshold_factor=2.5, smoothing_sigma=1):
    """
    Correct anomalies adaptively using local context.
    CRITICAL: Preserves 14-bit data range (0-16383) without clipping.
    """
    print("  - Applying adaptive contextual anomaly correction...")
    outliers_mask = adaptive_local_outlier_detection(image, block_size, threshold_factor)

    # Work with float64 for precision, then convert back
    resolved_image = image.astype(np.float64)

    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block_mask = outliers_mask[i:i+block_size, j:j+block_size]
            if np.any(block_mask):
                block = resolved_image[i:i+block_size, j:j+block_size]
                valid_pixels = block[~block_mask]
                if valid_pixels.size > 0:
                    replacement_value = np.mean(valid_pixels, dtype=np.float64)
                    block[block_mask] = replacement_value
                    resolved_image[i:i+block_size, j:j+block_size] = block

    # Apply Gaussian smoothing while preserving range
    smoothed = gaussian_filter(resolved_image, sigma=smoothing_sigma, mode='reflect')

    # Ensure no values exceed 14-bit range (0-16383)
    smoothed = np.clip(smoothed, 0, 16383)

    return smoothed.astype(np.uint16)


def detect_and_suppress_glare(image, intensity_threshold=14000):
    """
    Detect and suppress glare regions.
    CRITICAL: Maintains 14-bit integrity.
    Adjusted threshold for 14-bit range (max 16383).
    """
    print("  - Detecting and suppressing glare...")
    glare_mask = image > intensity_threshold
    labeled_mask = measure.label(glare_mask)
    image_suppressed = image.astype(np.float64)

    for region in measure.regionprops(labeled_mask):
        if region.area > 50:
            min_row, min_col, max_row, max_col = region.bbox
            region_slice = (slice(min_row, max_row), slice(min_col, max_col))
            # Suppress brightness by 30% while maintaining precision
            image_suppressed[region_slice][region.image] *= 0.7

    # Ensure values stay within 14-bit range
    image_suppressed = np.clip(image_suppressed, 0, 16383)

    return image_suppressed.astype(np.uint16)


def preprocess_thermal_image(thermal_image):
    """
    Complete preprocessing pipeline that preserves 14-bit data.
    Returns preprocessed image in uint16 format with 14-bit range.
    """
    print("  Starting preprocessing pipeline...")

    # Step 1: Adaptive contextual anomaly correction
    preprocessed = adaptive_contextual_anomaly_correction(
        thermal_image,
        block_size=30,
        threshold_factor=2.0,
        smoothing_sigma=1
    )

    # Step 2: Glare detection and suppression
    preprocessed = detect_and_suppress_glare(preprocessed, intensity_threshold=14000)

    print("  Preprocessing complete (14-bit integrity preserved)")
    return preprocessed


# ===========================
# COLORMAP APPLICATION
# ===========================

def apply_colormap_to_image(preprocessed_image, colormap_name):
    """
    Apply a colormap to a preprocessed 14-bit thermal image.
    Uses adaptive histogram equalization for better visualization.
    Returns RGB image in uint16 format.
    """
    # Normalize to 0-1 range for CLAHE (preserving relative values)
    normalized = preprocessed_image.astype(np.float64) / 16383.0

    # Apply CLAHE for better contrast
    enhanced = exposure.equalize_adapthist(normalized, clip_limit=0.03)

    # Apply colormap
    colormap = cm.get_cmap(colormap_name)
    colored_image = colormap(enhanced)[:, :, :3]  # Remove alpha channel

    # Convert to uint16 for high-quality output
    colored_image_uint16 = (colored_image * 65535).astype(np.uint16)

    return colored_image_uint16


# ===========================
# MAIN PROCESSING PIPELINE
# ===========================

def select_random_images(raw_dir, n=5):
    """
    Randomly select N TIFF images from the raw directory.
    """
    tiff_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.tif', '.tiff'))]

    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {raw_dir}")

    if len(tiff_files) < n:
        print(f"Warning: Only {len(tiff_files)} images available, using all of them.")
        return tiff_files

    selected = random.sample(tiff_files, n)
    print(f"\nRandomly selected {n} images:")
    for i, fname in enumerate(selected, 1):
        print(f"  {i}. {fname}")

    return selected


def process_images():
    """
    Main processing function:
    1. Randomly select N images from raw folder
    2. Preprocess and save to preprocessed folder (14-bit preserved)
    3. Apply all colormaps and save to separate folders
    """
    print("="*80)
    print("THERMAL IMAGE PREPROCESSING AND COLORMAP APPLICATION")
    print("="*80)

    # Create output directories
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(COLORMAP_BASE_DIR, exist_ok=True)

    # Select random images
    selected_images = select_random_images(RAW_DIR, N_IMAGES)

    # Process each selected image
    for idx, image_filename in enumerate(selected_images, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING IMAGE {idx}/{len(selected_images)}: {image_filename}")
        print(f"{'='*80}")

        image_path = os.path.join(RAW_DIR, image_filename)
        base_name = os.path.splitext(image_filename)[0]

        try:
            # Load raw thermal image
            print(f"\nLoading image: {image_path}")
            thermal_image = imread(image_path)

            if thermal_image is None or thermal_image.size == 0:
                print(f"ERROR: Could not read {image_path} or image is empty. Skipping.")
                continue

            print(f"  Image loaded successfully:")
            print(f"    Shape: {thermal_image.shape}")
            print(f"    Dtype: {thermal_image.dtype}")
            print(f"    Range: [{thermal_image.min()}, {thermal_image.max()}]")

            # Verify 14-bit data
            if thermal_image.max() > 16383:
                print(f"  WARNING: Image contains values > 14-bit range (16383)")

            # STEP 1: Preprocess and save
            print(f"\nSTEP 1: Preprocessing...")
            preprocessed_image = preprocess_thermal_image(thermal_image)

            preprocessed_path = os.path.join(PREPROCESSED_DIR, f"{base_name}_preprocessed.tiff")
            imwrite(preprocessed_path, preprocessed_image, compression='zlib')
            print(f"  Saved preprocessed image (14-bit): {preprocessed_path}")
            print(f"    Range: [{preprocessed_image.min()}, {preprocessed_image.max()}]")

            # STEP 2: Apply all colormaps
            print(f"\nSTEP 2: Applying colormaps ({len(COLORMAPS)} total)...")
            for cmap_idx, colormap_name in enumerate(COLORMAPS, 1):
                print(f"  [{cmap_idx}/{len(COLORMAPS)}] Applying colormap: {colormap_name}")

                # Create colormap-specific directory
                colormap_dir = os.path.join(COLORMAP_BASE_DIR, colormap_name)
                os.makedirs(colormap_dir, exist_ok=True)

                # Apply colormap
                colored_image = apply_colormap_to_image(preprocessed_image, colormap_name)

                # Save colormap output
                output_filename = f"{base_name}_{colormap_name}.tiff"
                output_path = os.path.join(colormap_dir, output_filename)
                imwrite(output_path, colored_image, compression='zlib')
                print(f"      Saved: {output_path}")

            print(f"\n  Completed processing for {image_filename}")

        except Exception as e:
            print(f"\nERROR processing {image_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Processed images: {len(selected_images)}")
    print(f"  - Preprocessed images saved to: {PREPROCESSED_DIR}")
    print(f"  - Colormap outputs saved to: {COLORMAP_BASE_DIR}")
    print(f"  - Total colormaps applied: {len(COLORMAPS)}")
    print(f"  - Total colormap images generated: {len(selected_images) * len(COLORMAPS)}")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional - remove for true randomness)
    random.seed(42)

    process_images()

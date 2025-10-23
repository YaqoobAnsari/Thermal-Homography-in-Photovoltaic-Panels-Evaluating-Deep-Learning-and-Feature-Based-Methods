import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure, measure
import os

# Enhanced adaptive local outlier detection
def improved_adaptive_local_outlier_detection(image, block_size=50, threshold_factor=2.5):
    outliers_mask = np.zeros_like(image, dtype=bool)
    height, width = image.shape
    padded_image = np.pad(image, block_size // 2, mode='reflect')

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = padded_image[i:i + block_size * 2, j:j + block_size * 2]
            local_mean = np.mean(block)
            local_std_dev = np.std(block)
            local_threshold_high = local_mean + threshold_factor * local_std_dev
            local_threshold_low = local_mean - threshold_factor * local_std_dev
            block_outliers = (image[i:i + block_size, j:j + block_size] > local_threshold_high) | (
                image[i:i + block_size, j:j + block_size] < local_threshold_low
            )
            outliers_mask[i:i + block_size, j:j + block_size] = block_outliers

    return outliers_mask

# Enhanced glare detection and suppression
def improved_glare_detection(image, intensity_threshold_ratio=0.98, min_area=50):
    dynamic_threshold = np.percentile(image, intensity_threshold_ratio * 100)
    glare_mask = image > dynamic_threshold
    labeled_mask = measure.label(glare_mask)
    image_suppressed = np.copy(image)

    for region in measure.regionprops(labeled_mask):
        if region.area > min_area:
            min_row, min_col, max_row, max_col = region.bbox
            region_slice = (slice(min_row, max_row), slice(min_col, max_col))
            mean_intensity = np.mean(image[region_slice][region.image])
            image_suppressed[region_slice][region.image] = (
                mean_intensity + 0.3 * (dynamic_threshold - mean_intensity)
            )

    return image_suppressed

# Adaptive Contextual Anomaly Correction (ACAC)
def adaptive_contextual_anomaly_correction(image, block_size=50, threshold_factor=2.5, smoothing_sigma=1):
    outliers_mask = improved_adaptive_local_outlier_detection(image, block_size, threshold_factor)
    resolved_image = np.copy(image)

    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            mask_block = outliers_mask[i:i+block_size, j:j+block_size]

            if np.any(mask_block):
                non_outlier_pixels = block[~mask_block]
                if non_outlier_pixels.size > 0:
                    replacement_value = np.mean(non_outlier_pixels)
                    block[mask_block] = replacement_value
                resolved_image[i:i+block_size, j:j+block_size] = block

    resolved_image = gaussian_filter(resolved_image, sigma=smoothing_sigma)
    return resolved_image, outliers_mask

# Full preprocessing pipeline
def improved_preprocess_image(image_path, output_folder, save_outlier_mask=False):
    # Load the 16-bit image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Preserve dynamic range without scaling it down for visualization
    image_dynamic = np.clip(image, 0, 65535)
    
    # Print original statistics
    print(f"Original Image Stats - Min: {np.min(image_dynamic)}, Max: {np.max(image_dynamic)}, Avg: {np.mean(image_dynamic)}")

    # Improved ACAC method
    resolved_image_aggressive, adaptive_outliers_mask = adaptive_contextual_anomaly_correction(
        image_dynamic, block_size=10, threshold_factor=4.0, smoothing_sigma=0.2
    )

    # Improved glare suppression
    resolved_image_suppressed = improved_glare_detection(resolved_image_aggressive)

    # Apply noise-aware histogram equalization
    image_suppressed_smoothed = gaussian_filter(resolved_image_suppressed, sigma=0.5)
    image_enhanced = exposure.equalize_adapthist(image_suppressed_smoothed, clip_limit=0.02)
    image_enhanced_normalized = (image_enhanced * 65535).astype(np.uint16)
    
    # Print post-processed statistics
    print(f"Processed Image Stats - Min: {np.min(image_enhanced_normalized)}, Max: {np.max(image_enhanced_normalized)}, Avg: {np.mean(image_enhanced_normalized)}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the raw input image
    base_name = os.path.basename(image_path)
    base, ext = os.path.splitext(base_name)
    raw_image_path = os.path.join(output_folder, f"{base}_raw_improved{ext}")
    cv2.imwrite(raw_image_path, image_dynamic)

    # Save the preprocessed image
    preprocessed_image_path = os.path.join(output_folder, f"{base}_preprocessed_improved{ext}")
    cv2.imwrite(preprocessed_image_path, image_enhanced_normalized)

    # Optionally save the outlier mask
    if save_outlier_mask:
        outlier_mask = np.zeros_like(image, dtype=np.uint16)
        outlier_mask[adaptive_outliers_mask] = 65535
        outlier_mask_path = os.path.join(output_folder, f"{base}_outlier_mask_improved{ext}")
        cv2.imwrite(outlier_mask_path, outlier_mask)

    return raw_image_path, preprocessed_image_path, outlier_mask_path if save_outlier_mask else None
 
# Example usage
image_path = '20241210_123846/20241224_135940_858_IR.TIFF'  # Update to your file path
output_folder = 'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Playground'
improved_preprocess_image(image_path, output_folder, save_outlier_mask=True)

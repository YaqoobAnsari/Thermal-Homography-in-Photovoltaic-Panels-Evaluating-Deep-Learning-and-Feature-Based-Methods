import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure, measure
import os

# Function to detect outliers using localized adaptive thresholds
def adaptive_local_outlier_detection(image, block_size=50, threshold_factor=2.5):
    outliers_mask = np.zeros_like(image, dtype=bool)
    height, width = image.shape

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            local_mean = np.mean(block)
            local_std_dev = np.std(block)
            local_threshold_high = local_mean + threshold_factor * local_std_dev
            local_threshold_low = local_mean - threshold_factor * local_std_dev
            block_outliers = (block > local_threshold_high) | (block < local_threshold_low)
            outliers_mask[i:i+block_size, j:j+block_size] = block_outliers

    return outliers_mask

# Function to apply the ACAC method for outlier detection and correction
def adaptive_contextual_anomaly_correction(image, block_size=50, threshold_factor=2.5, smoothing_sigma=1):
    outliers_mask = adaptive_local_outlier_detection(image, block_size, threshold_factor)
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

# Function to detect and suppress bright glare regions
def detect_and_suppress_glare(image, intensity_threshold=60000):
    glare_mask = image > intensity_threshold
    labeled_mask = measure.label(glare_mask)
    image_suppressed = np.copy(image)
    
    for region in measure.regionprops(labeled_mask):
        if region.area > 50:  # Filter out small noise
            min_row, min_col, max_row, max_col = region.bbox
            region_slice = (slice(min_row, max_row), slice(min_col, max_col))
            image_suppressed[region_slice][region.image] = (
                image_suppressed[region_slice][region.image] * 0.7  # Suppress brightness
            )

    return image_suppressed

# Main preprocessing function
def preprocess_image(image_path, output_folder, save_outlier_mask=False):
    # Load the 16-bit image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_normalized = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX)

    # Apply the ACAC method with aggressive parameters
    resolved_image_aggressive, adaptive_outliers_mask = adaptive_contextual_anomaly_correction(
        image, block_size=10, threshold_factor=4.0, smoothing_sigma=0.2
    )

    # Apply glare detection and suppression
    resolved_image_suppressed = detect_and_suppress_glare(resolved_image_aggressive)

    # Apply adaptive histogram equalization for better detail enhancement
    image_enhanced = exposure.equalize_adapthist(resolved_image_suppressed, clip_limit=0.03)
    image_enhanced_normalized = (image_enhanced * 65535).astype(np.uint16)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the raw input image
    base_name = os.path.basename(image_path)
    base, ext = os.path.splitext(base_name)
    raw_image_path = os.path.join(output_folder, f"{base}_raw{ext}")
    cv2.imwrite(raw_image_path, image_normalized)
    print(f"Raw input image saved to: {raw_image_path}")

    # Save the preprocessed image
    preprocessed_image_path = os.path.join(output_folder, f"{base}_preprocessed{ext}")
    cv2.imwrite(preprocessed_image_path, image_enhanced_normalized)
    print(f"Preprocessed image saved to: {preprocessed_image_path}")

    # Optionally save the outlier mask
    if save_outlier_mask:
        outlier_mask = np.zeros_like(image, dtype=np.uint16)
        outlier_mask[adaptive_outliers_mask] = 65535
        outlier_mask_path = os.path.join(output_folder, f"{base}_outlier_mask{ext}")
        cv2.imwrite(outlier_mask_path, outlier_mask)
        print(f"Outlier mask saved to: {outlier_mask_path}")

# Example usage
image_path = '20241210_123846/20241224_135940_858_IR.TIFF'
output_folder = 'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Playground'
preprocess_image(image_path, output_folder, save_outlier_mask=True)

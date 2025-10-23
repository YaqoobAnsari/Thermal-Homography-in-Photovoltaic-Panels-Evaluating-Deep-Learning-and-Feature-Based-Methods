import os
import numpy as np
from matplotlib import cm
from tifffile import imread, imsave
from scipy.ndimage import gaussian_filter
from skimage import exposure, measure

# Colormaps for comprehensive evaluation
colormaps = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'copper', 'Greys', 'Blues',
    'Oranges', 'Reds', 'coolwarm', 'seismic', 'RdBu', 'BrBG', 'PiYG', 'twilight', 'twilight_shifted',
    'hsv', 'tab10', 'Set1', 'Set2', 'Pastel1', 'terrain', 'nipy_spectral', 'flag'
]

# Preprocessing functions
def adaptive_local_outlier_detection(image, block_size=50, threshold_factor=2.5):
    outliers_mask = np.zeros_like(image, dtype=bool)
    height, width = image.shape
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            local_mean, local_std_dev = np.mean(block), np.std(block)
            high, low = local_mean + threshold_factor * local_std_dev, local_mean - threshold_factor * local_std_dev
            outliers_mask[i:i+block_size, j:j+block_size] = (block > high) | (block < low)
    return outliers_mask

def adaptive_contextual_anomaly_correction(image, block_size=50, threshold_factor=2.5, smoothing_sigma=1):
    print("Applying adaptive contextual anomaly correction...")
    outliers_mask = adaptive_local_outlier_detection(image, block_size, threshold_factor)
    resolved_image = np.copy(image)
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            if np.any(outliers_mask[i:i+block_size, j:j+block_size]):
                replacement_value = np.mean(block[~outliers_mask[i:i+block_size, j:j+block_size]])
                block[outliers_mask[i:i+block_size, j:j+block_size]] = replacement_value
            resolved_image[i:i+block_size, j:j+block_size] = block
    return gaussian_filter(resolved_image, sigma=smoothing_sigma)

def detect_and_suppress_glare(image, intensity_threshold=60000):
    print("Detecting and suppressing glare...")
    glare_mask = image > intensity_threshold
    labeled_mask = measure.label(glare_mask)
    image_suppressed = np.copy(image)
    for region in measure.regionprops(labeled_mask):
        if region.area > 50:
            min_row, min_col, max_row, max_col = region.bbox
            region_slice = (slice(min_row, max_row), slice(min_col, max_col))
            image_suppressed[region_slice][region.image] *= 0.7  # Suppress brightness
    return image_suppressed

# Main function for colormap application
def create_colormap_datasets(source_dir, base_output_dir):
    for colormap_name in colormaps:
        print(f"\nApplying colormap: {colormap_name}")
        colormap_dir = os.path.join(base_output_dir, f'Dataset June2024 {colormap_name.capitalize()}')
        os.makedirs(colormap_dir, exist_ok=True)
        print(f"Created colormap directory: {colormap_dir}")
        
        for root, dirs, files in os.walk(source_dir):
            for directory in dirs:
                dir_path = os.path.join(colormap_dir, os.path.relpath(os.path.join(root, directory), source_dir))
                os.makedirs(dir_path, exist_ok=True)
                print(f"Ensured directory exists: {dir_path}")

            for file in files:
                if file.lower().endswith('.tif'):
                    image_path = os.path.join(root, file)
                    print(f"\nProcessing image: {image_path}")
                    try:
                        thermal_image = imread(image_path)
                        if thermal_image is None or thermal_image.size == 0:
                            print(f"Warning: {image_path} could not be read or is empty.")
                            continue
                        else:
                            print(f"Image {image_path} read successfully with shape {thermal_image.shape}.")

                        # Apply preprocessing steps
                        print("Starting preprocessing...")
                        preprocessed_image = adaptive_contextual_anomaly_correction(thermal_image, block_size=30, threshold_factor=2.0, smoothing_sigma=1)
                        preprocessed_image = detect_and_suppress_glare(preprocessed_image)
                        enhanced_image = exposure.equalize_adapthist(preprocessed_image, clip_limit=0.03)
                        enhanced_image_normalized = (enhanced_image * 65535).astype(np.uint16)
                        print("Preprocessing complete.")

                        # Apply colormap
                        colormap = cm.get_cmap(colormap_name)
                        colored_image = (colormap(enhanced_image_normalized / 65535)[:, :, :3] * 65535).astype(np.uint16)
                        print(f"Applied colormap {colormap_name} to image.")

                        # Define and save to output path
                        output_path = os.path.join(
                            colormap_dir, os.path.relpath(root, source_dir), file.replace('.tif', f'_{colormap_name}.png')
                        )
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        imsave(output_path, colored_image)
                        print(f"Saved image with {colormap_name} colormap to {output_path}")

                    except Exception as e:
                        print(f"Error processing {image_path} with colormap {colormap_name}: {e}")

# Run the colormap application process
source_thermal_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset\Dataset June2024 Thermal'
base_output_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset'
create_colormap_datasets(source_thermal_dir, base_output_dir)
print("All colormap datasets created successfully!")

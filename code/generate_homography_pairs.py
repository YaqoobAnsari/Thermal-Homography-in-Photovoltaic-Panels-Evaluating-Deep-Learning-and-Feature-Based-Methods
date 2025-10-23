"""
generate_homography_pairs.py

Homography pair generator for thermal image benchmarking.
Processes all 29 data types: raw, preprocessed, and 27 colormap variants.

Generates synthetic homography training pairs with proper geometric transformations.
Adapted from datagen.py with multi-dataset support and relative paths.

Author: Homography Benchmarking Project
Date: 2025
"""

import cv2
import numpy as np
import random
import os
import math
from tifffile import imread

# Get the project root directory (parent of 'code' folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Configuration
IMAGE_DATA_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Image Data')
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, 'benchmarking_dataset', 'Homography Pairs')

# Data types to process
DATA_TYPES = {
    'raw': os.path.join(IMAGE_DATA_DIR, 'raw'),
    'preprocessed': os.path.join(IMAGE_DATA_DIR, 'preprocessed'),
}

# Add all 27 colormaps
COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'copper', 'Greys', 'Blues',
    'Oranges', 'Reds', 'coolwarm', 'seismic', 'RdBu', 'BrBG', 'PiYG', 'twilight', 'twilight_shifted',
    'hsv', 'tab10', 'Set1', 'Set2', 'Pastel1', 'terrain', 'nipy_spectral', 'flag'
]

for cmap in COLORMAPS:
    DATA_TYPES[f'colormap_{cmap}'] = os.path.join(IMAGE_DATA_DIR, 'colormap_outputs', cmap)


# ============================================================================
# PATCH GENERATION FUNCTIONS
# ============================================================================

def generate_patch(image, patch_size, center_bias_factor=0.2):
    """
    Generate initial random patch coordinates with center bias.

    Args:
        image: Input image
        patch_size: (width, height) of patch
        center_bias_factor: Bias towards center (0-1)

    Returns:
        Tuple of (top_left_coords, four_corner_points)
    """
    h, w = image.shape[:2]
    max_x = w - patch_size[0]
    max_y = h - patch_size[1]

    if max_x < 0 or max_y < 0:
        raise ValueError(f"Patch size {patch_size} is larger than image size {(w, h)}")

    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2

    # Standard deviation for the Gaussian distribution
    std_x, std_y = int(w * center_bias_factor), int(h * center_bias_factor)

    # Generate x and y coordinates with a bias towards the center
    x = int(np.clip(np.random.normal(center_x, std_x), 0, max_x))
    y = int(np.clip(np.random.normal(center_y, std_y), 0, max_y))

    # Generate the four corner points of the rectangular patch
    four_points = [
        (x, y),                                    # Top-left
        (x + patch_size[0], y),                    # Top-right
        (x + patch_size[0], y + patch_size[1]),    # Bottom-right
        (x, y + patch_size[1])                     # Bottom-left
    ]

    return (x, y), four_points


def perturb_corners(four_points, rho, image_shape):
    """
    Perturb corner points by random offsets within [-rho, +rho].

    Args:
        four_points: List of 4 corner coordinates
        rho: Maximum perturbation distance in pixels
        image_shape: Shape of the image (h, w) or (h, w, c)

    Returns:
        List of 4 perturbed corner coordinates
    """
    h, w = image_shape[:2]
    perturbed_points = []

    for point in four_points:
        # Keep trying until we get a valid perturbation
        max_attempts = 100
        for attempt in range(max_attempts):
            delta_x = random.randint(-rho, rho)
            delta_y = random.randint(-rho, rho)
            perturbed_x = point[0] + delta_x
            perturbed_y = point[1] + delta_y

            # Ensure the perturbed point is within image bounds
            if 0 <= perturbed_x < w and 0 <= perturbed_y < h:
                perturbed_points.append((perturbed_x, perturbed_y))
                break
        else:
            # If we couldn't find valid perturbation, use original point
            perturbed_points.append(point)

    return perturbed_points


def slide_patch(image, initial_coords, patch_size, overlap):
    """
    Slide patch to a new position with specified overlap.

    Args:
        image: Input image
        initial_coords: (x, y) of initial patch
        patch_size: (width, height) of patch
        overlap: Overlap fraction (0-1)

    Returns:
        Tuple of (new_coords, new_four_points)
    """
    x, y = initial_coords
    h, w = image.shape[:2]

    directions = ["right", "left", "down", "up", "down-right", "down-left", "up-right", "up-left"]
    slide_successful = False

    while not slide_successful and directions:
        direction = random.choice(directions)
        slide_x, slide_y = 0, 0

        if direction == "right":
            slide_x = int(patch_size[0] * (1 - overlap))
            new_x, new_y = x + slide_x, y
            if new_x + patch_size[0] <= w:
                slide_successful = True
        elif direction == "left":
            slide_x = int(patch_size[0] * (1 - overlap))
            new_x, new_y = x - slide_x, y
            if new_x >= 0:
                slide_successful = True
        elif direction == "down":
            slide_y = int(patch_size[1] * (1 - overlap))
            new_x, new_y = x, y + slide_y
            if new_y + patch_size[1] <= h:
                slide_successful = True
        elif direction == "up":
            slide_y = int(patch_size[1] * (1 - overlap))
            new_x, new_y = x, y - slide_y
            if new_y >= 0:
                slide_successful = True
        elif direction == "down-right":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))
            new_x, new_y = x + slide_x, y + slide_y
            if new_x + patch_size[0] <= w and new_y + patch_size[1] <= h:
                slide_successful = True
        elif direction == "down-left":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))
            new_x, new_y = x - slide_x, y + slide_y
            if new_x >= 0 and new_y + patch_size[1] <= h:
                slide_successful = True
        elif direction == "up-right":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))
            new_x, new_y = x + slide_x, y - slide_y
            if new_x + patch_size[0] <= w and new_y >= 0:
                slide_successful = True
        elif direction == "up-left":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))
            new_x, new_y = x - slide_x, y - slide_y
            if new_x >= 0 and new_y >= 0:
                slide_successful = True

        if not slide_successful:
            directions.remove(direction)

    if not slide_successful:
        raise ValueError("Unable to slide in any direction without exceeding image boundaries.")

    # Generate the new four corner points for the slid patch
    new_four_points = [
        (new_x, new_y),
        (new_x + patch_size[0], new_y),
        (new_x + patch_size[0], new_y + patch_size[1]),
        (new_x, new_y + patch_size[1])
    ]

    return (new_x, new_y), new_four_points


# ============================================================================
# HOMOGRAPHY COMPUTATION
# ============================================================================

def compute_homography_matrix(patch1_corners, patch2_corners, apply_perturbation=True, rho=0, image_shape=None):
    """
    Compute homography matrix that maps patch1 to patch2.

    Args:
        patch1_corners: 4 corner points of patch 1
        patch2_corners: 4 corner points of patch 2 (after sliding)
        apply_perturbation: Whether to perturb patch2 corners
        rho: Perturbation magnitude
        image_shape: Image shape for boundary checking

    Returns:
        Tuple of (H, patch2_corners_final, delta_4pt)
    """
    if apply_perturbation and rho > 0 and image_shape is not None:
        patch2_corners_perturbed = perturb_corners(patch2_corners, rho, image_shape)
    else:
        patch2_corners_perturbed = patch2_corners

    # Compute homography: patch1 -> patch2_perturbed
    src_points = np.float32(patch1_corners)
    dst_points = np.float32(patch2_corners_perturbed)

    H = cv2.getPerspectiveTransform(src_points, dst_points)

    # Compute 4-point offset representation
    delta_4pt = np.array(patch2_corners_perturbed) - np.array(patch1_corners)

    return H, patch2_corners_perturbed, delta_4pt


def verify_homography(H, patch1_corners, patch2_corners_expected, tolerance=2.0):
    """
    Verify that homography H correctly maps patch1_corners to patch2_corners_expected.

    Returns:
        Tuple of (is_valid, max_error, mean_error)
    """
    src_points = np.float32([patch1_corners])
    transformed_points = cv2.perspectiveTransform(src_points, H)[0]

    errors = np.linalg.norm(transformed_points - np.array(patch2_corners_expected), axis=1)
    max_error = np.max(errors)
    mean_error = np.mean(errors)

    is_valid = max_error < tolerance

    return is_valid, max_error, mean_error


# ============================================================================
# IMAGE HANDLING
# ============================================================================

def load_image(image_path):
    """
    Load image, handling both TIFF and standard formats.
    For 14-bit thermal images: applies histogram stretching and CLAHE before converting to uint8.
    """
    if image_path.lower().endswith(('.tif', '.tiff')):
        # Use tifffile for TIFF images
        img = imread(image_path)

        # If uint16 (14-bit thermal data), process properly before converting to uint8
        if img.dtype == np.uint16:
            # Check if grayscale (thermal) or RGB (colormap)
            if len(img.shape) == 2:  # Grayscale thermal image
                # Step 1: Histogram stretching on 14-bit data (0-16383 range)
                # Use percentile-based stretching to handle outliers
                p2, p98 = np.percentile(img, (2, 98))
                img_stretched = np.clip((img.astype(np.float64) - p2) / (p98 - p2) * 16383, 0, 16383).astype(np.uint16)

                # Step 2: Convert to uint8 for CLAHE (CLAHE only works on 8-bit)
                img_8bit = (img_stretched / 256).astype(np.uint8)  # Divide by 256 to map 14-bit to 8-bit

                # Step 3: Apply CLAHE on 8-bit data to enhance local contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                img_enhanced = clahe.apply(img_8bit)

                return img_enhanced
            else:  # RGB colormap image (already 3-channel)
                # Just normalize to uint8 range for colormaps
                img_8bit = (img / 256).astype(np.uint8)
                return img_8bit
        return img
    else:
        # Use OpenCV for other formats (colormaps are already RGB)
        return cv2.imread(image_path)


def extract_patch_from_corners(image, corners):
    """
    Extract rectangular patch from image given 4 corners.
    """
    corners_array = np.array(corners, dtype=np.int32)
    x_min = max(0, int(np.min(corners_array[:, 0])))
    y_min = max(0, int(np.min(corners_array[:, 1])))
    x_max = min(image.shape[1], int(np.max(corners_array[:, 0])))
    y_max = min(image.shape[0], int(np.max(corners_array[:, 1])))

    patch = image[y_min:y_max, x_min:x_max]
    return patch


def check_patch_validity(image, corners, padding_threshold=0.05):
    """
    Check if patch contains excessive black padding or is out of bounds.
    """
    h, w = image.shape[:2]

    # Check if any corner is out of bounds
    for x, y in corners:
        if x < 0 or x >= w or y < 0 or y >= h:
            return False

    # Extract patch
    patch = extract_patch_from_corners(image, corners)

    if patch.size == 0:
        return False

    # Count black pixels
    if len(patch.shape) == 3:
        black_pixels = np.all(patch == 0, axis=2)
    else:
        black_pixels = (patch == 0)

    black_ratio = np.sum(black_pixels) / patch.size

    return black_ratio < padding_threshold


# ============================================================================
# METADATA SAVING
# ============================================================================

def save_metadata(sample_dir, image_path, data_type, patch_size, overlap, rho,
                  patch1_corners, patch2_corners, patch2_corners_perturbed,
                  H, delta_4pt, verification_error):
    """
    Save comprehensive metadata for the sample.
    """
    metadata_path = os.path.join(sample_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Data Type: {data_type}\n")
        f.write(f"Image Path: {os.path.abspath(image_path)}\n")
        f.write(f"Image Name: {os.path.basename(image_path)}\n")
        f.write(f"Patch Size: {patch_size}\n")
        f.write(f"Overlap: {overlap:.4f}\n")
        f.write(f"Rho (Perturbation): {rho}\n")
        f.write(f"\n")
        f.write(f"Patch 1 Corners: {patch1_corners}\n")
        f.write(f"Patch 2 Corners (Original): {patch2_corners}\n")
        f.write(f"Patch 2 Corners (Perturbed): {patch2_corners_perturbed}\n")
        f.write(f"\n")
        f.write(f"Homography Verification:\n")
        f.write(f"  Max Error: {verification_error[0]:.4f} pixels\n")
        f.write(f"  Mean Error: {verification_error[1]:.4f} pixels\n")
        f.write(f"  Valid: {verification_error[2]}\n")


# ============================================================================
# MAIN SAMPLE GENERATION
# ============================================================================

def generate_samples_for_image(image_path, data_type, output_dir, num_samples=10,
                               patch_size=(256, 256), overlap_range=(0.25, 0.55),
                               rho_range=(15, 50), validate_homography=True):
    """
    Generate homography training samples from a single image.

    Args:
        image_path: Path to source image
        data_type: Type of data (raw, preprocessed, colormap_xxx)
        output_dir: Output directory for this image
        num_samples: Number of samples to generate
        patch_size: (width, height) of patches
        overlap_range: (min, max) overlap fraction
        rho_range: (min, max) perturbation distance
        validate_homography: Whether to validate H correctness

    Returns:
        Number of successfully generated samples
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"    ERROR: Unable to load image at {image_path}")
        return 0

    h, w = image.shape[:2]

    success_count = 0
    attempt = 0
    max_attempts = num_samples * 3

    while success_count < num_samples and attempt < max_attempts:
        attempt += 1

        try:
            # Random parameters
            overlap = random.uniform(overlap_range[0], overlap_range[1])
            rho = random.randint(rho_range[0], rho_range[1])

            # Generate patch 1
            patch1_coords, patch1_corners = generate_patch(image, patch_size, center_bias_factor=0.2)

            # Slide to patch 2
            patch2_coords, patch2_corners = slide_patch(image, patch1_coords, patch_size, overlap)

            # Compute homography with perturbation
            H, patch2_corners_perturbed, delta_4pt = compute_homography_matrix(
                patch1_corners, patch2_corners,
                apply_perturbation=True, rho=rho, image_shape=image.shape
            )

            # Validate homography
            if validate_homography:
                is_valid, max_error, mean_error = verify_homography(
                    H, patch1_corners, patch2_corners_perturbed, tolerance=2.0
                )
                if not is_valid:
                    continue
            else:
                max_error, mean_error = 0.0, 0.0
                is_valid = True

            # Check patch validity
            if not check_patch_validity(image, patch1_corners):
                continue

            if not check_patch_validity(image, patch2_corners_perturbed):
                continue

            # Extract patches
            patch_A = extract_patch_from_corners(image, patch1_corners)
            patch_B = extract_patch_from_corners(image, patch2_corners_perturbed)

            if patch_A.size == 0 or patch_B.size == 0:
                continue

            # Save sample
            success_count += 1
            sample_dir = os.path.join(output_dir, f'pair_{success_count:04d}')
            os.makedirs(sample_dir, exist_ok=True)

            # Save patches directly (CLAHE already applied during load_image for thermal data)
            cv2.imwrite(os.path.join(sample_dir, 'patch_A.png'), patch_A)
            cv2.imwrite(os.path.join(sample_dir, 'patch_B.png'), patch_B)

            # Save homography matrix
            np.savetxt(os.path.join(sample_dir, 'homography_H.txt'), H, fmt='%.8f')

            # Save 4-point offset
            np.savetxt(os.path.join(sample_dir, 'delta_4pt.txt'), delta_4pt, fmt='%.8f')

            # Save metadata
            save_metadata(sample_dir, image_path, data_type, patch_size, overlap, rho,
                         patch1_corners, patch2_corners, patch2_corners_perturbed,
                         H, delta_4pt, (max_error, mean_error, is_valid))

        except Exception as e:
            continue

    return success_count


def process_data_type(data_type, data_dir, num_samples_per_image=10,
                      patch_size=(256, 256)):
    """
    Process all images in a data type directory.

    Args:
        data_type: Name of data type (e.g., 'raw', 'preprocessed', 'colormap_viridis')
        data_dir: Directory containing images
        num_samples_per_image: Number of homography pairs per image
        patch_size: Patch size for homography generation

    Returns:
        Total number of samples generated
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING DATA TYPE: {data_type}")
    print(f"{'='*80}")

    if not os.path.exists(data_dir):
        print(f"  ERROR: Directory not found: {data_dir}")
        return 0

    # Get all image files
    image_files = [f for f in os.listdir(data_dir)
                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]

    if len(image_files) == 0:
        print(f"  WARNING: No images found in {data_dir}")
        return 0

    print(f"  Found {len(image_files)} images")

    # Create output directory for this data type
    output_dir = os.path.join(OUTPUT_BASE_DIR, data_type)
    os.makedirs(output_dir, exist_ok=True)

    total_samples = 0

    for idx, img_file in enumerate(image_files, 1):
        image_path = os.path.join(data_dir, img_file)
        base_name = os.path.splitext(img_file)[0]

        print(f"\n  [{idx}/{len(image_files)}] Processing: {img_file}")

        # Create output directory for this image
        image_output_dir = os.path.join(output_dir, base_name)

        # Generate samples
        num_generated = generate_samples_for_image(
            image_path=image_path,
            data_type=data_type,
            output_dir=image_output_dir,
            num_samples=num_samples_per_image,
            patch_size=patch_size,
            overlap_range=(0.25, 0.55),
            rho_range=(15, 50),
            validate_homography=True
        )

        total_samples += num_generated
        print(f"    Generated {num_generated}/{num_samples_per_image} pairs")

    print(f"\n  DATA TYPE SUMMARY: {total_samples} total pairs generated")
    return total_samples


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Process all 29 data types and generate homography pairs.
    """
    print("="*80)
    print("HOMOGRAPHY PAIR GENERATION FOR BENCHMARKING DATASET")
    print("="*80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Image Data Directory: {IMAGE_DATA_DIR}")
    print(f"Output Directory: {OUTPUT_BASE_DIR}")
    print(f"\nTotal Data Types: {len(DATA_TYPES)}")

    # Configuration
    NUM_SAMPLES_PER_IMAGE = 10  # Number of pairs per image
    PATCH_SIZE = (256, 256)     # Patch size for homography

    print(f"Samples per image: {NUM_SAMPLES_PER_IMAGE}")
    print(f"Patch size: {PATCH_SIZE}")

    # Process each data type
    grand_total = 0
    results = {}

    for data_type, data_dir in DATA_TYPES.items():
        total = process_data_type(
            data_type=data_type,
            data_dir=data_dir,
            num_samples_per_image=NUM_SAMPLES_PER_IMAGE,
            patch_size=PATCH_SIZE
        )
        results[data_type] = total
        grand_total += total

    # Final summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE - SUMMARY")
    print("="*80)

    for data_type, count in results.items():
        print(f"  {data_type:30s}: {count:5d} pairs")

    print(f"\n  {'TOTAL':30s}: {grand_total:5d} pairs")
    print(f"\nOutput location: {OUTPUT_BASE_DIR}")
    print("="*80)


if __name__ == "__main__":
    # Set random seed for reproducibility (optional - remove for true randomness)
    random.seed(42)
    np.random.seed(42)

    main()

import cv2
import numpy as np
import random
import os
import math
from Traditional_model_12 import compute_metrics_for_distance_folders

def verify_directory_structure(colormap_transformed_dataset, raw_thermal_dataset):
    # Expected folder and file structure requirements
    expected_colormap_count = 27
    timing_folders = {"June02_12pm", "June05_8am", "June09_12pm"}
    
    def has_required_images(folder_path, file_extension=".png"):
        """Check if a folder contains at least one image file with the specified extension."""
        return any(f.lower().endswith(file_extension) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)))
    
    def check_final_folders_have_images(base_path, file_extension=".png"):
        """Traverse directory structure to ensure no end folder is empty of specified images."""
        for root, dirs, files in os.walk(base_path):
            if not dirs:  # No subdirectories, so this should be a final folder
                if not has_required_images(root, file_extension):
                    print(f"Error: Folder '{root}' does not contain any {file_extension.upper()} images.")
                    return False
        return True

    # 1. Check that we have 27 colormap folders in colormap_transformed_dataset
    colormap_folders = [f for f in os.listdir(colormap_transformed_dataset) if os.path.isdir(os.path.join(colormap_transformed_dataset, f))]
    if len(colormap_folders) != expected_colormap_count:
        print(f"Expected {expected_colormap_count} colormap folders, but found {len(colormap_folders)}.")
        return False
    
    # 2. Verify timing folders and final structure in each colormap folder
    for colormap_folder in colormap_folders:
        colormap_path = os.path.join(colormap_transformed_dataset, colormap_folder)
        for timing_folder in timing_folders:
            timing_path = os.path.join(colormap_path, timing_folder)
            if not os.path.isdir(timing_path):
                print(f"Missing timing folder '{timing_folder}' in {colormap_path}")
                return False
            if not check_final_folders_have_images(timing_path, file_extension=".png"):
                print(f"Invalid structure or empty folders found in '{timing_path}'")
                return False

    # 3. Verify timing folders and final structure in raw thermal dataset, using .tif files
    for timing_folder in timing_folders:
        timing_path = os.path.join(raw_thermal_dataset, timing_folder)
        if not os.path.isdir(timing_path):
            print(f"Missing timing folder '{timing_folder}' in raw thermal dataset at {raw_thermal_dataset}")
            return False
        if not check_final_folders_have_images(timing_path, file_extension=".tif"):
            print(f"Invalid structure or empty folders found in '{timing_path}'")
            return False

    print("All directory structures verified successfully.")
    return True


def create_generated_dataset(colormap_transformed_dataset, raw_thermal_dataset, output_base_dir):
    # Define the output directory for the Generated Dataset
    generated_dataset_dir = os.path.join(output_base_dir, "Generated Dataset")
    os.makedirs(generated_dataset_dir, exist_ok=True)
    
    # 1. Create colormap subfolders in Generated Dataset, each mimicking its directory structure
    colormap_folders = [f for f in os.listdir(colormap_transformed_dataset) if os.path.isdir(os.path.join(colormap_transformed_dataset, f))]
    
    for colormap_folder in colormap_folders:
        # Extract the colormap name by removing the prefix "Dataset June2024 " from the folder name
        colormap_name = colormap_folder.replace("Dataset June2024 ", "")
        colormap_generated_dir = os.path.join(generated_dataset_dir, colormap_name)
        os.makedirs(colormap_generated_dir, exist_ok=True)
        
        # Mimic the directory structure within each colormap folder
        colormap_source_dir = os.path.join(colormap_transformed_dataset, colormap_folder)
        for root, dirs, _ in os.walk(colormap_source_dir):
            relative_path = os.path.relpath(root, colormap_source_dir)
            for dir_name in dirs:
                target_dir = os.path.join(colormap_generated_dir, relative_path, dir_name)
                os.makedirs(target_dir, exist_ok=True)
    
    # 2. Create the Raw Thermal folder in Generated Dataset
    raw_thermal_generated_dir = os.path.join(generated_dataset_dir, "Raw Thermal")
    os.makedirs(raw_thermal_generated_dir, exist_ok=True)
    
    # Mimic the directory structure within the raw thermal dataset
    for root, dirs, _ in os.walk(raw_thermal_dataset):
        relative_path = os.path.relpath(root, raw_thermal_dataset)
        for dir_name in dirs:
            target_dir = os.path.join(raw_thermal_generated_dir, relative_path, dir_name)
            os.makedirs(target_dir, exist_ok=True)
    
    print("Generated Dataset folder structure created successfully.")

def visualize_patches(img, box_coords_1, box_coords_2, perturbed_points, patch_size, save_path='visualized_patches.png', show_perturbed=True, thickness=2):
    # Colors for the original and translated boxes
    color_original = (0, 255, 0)  # Green for original bounding box
    color_translated = (0, 0, 255)  # Red for translated bounding box
    color_perturbed = (255, 0, 255)  # Purple for perturbed points

    # Visualize the first patch (original rectangle)
    x1, y1 = box_coords_1
    x1_end, y1_end = x1 + patch_size[0], y1 + patch_size[1]
    cv2.rectangle(img, (x1, y1), (x1_end, y1_end), color_original, thickness)
    cv2.putText(img, 'Original', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_original, 2)

    # Visualize the second patch (translated rectangle)
    x2, y2 = box_coords_2
    x2_end, y2_end = x2 + patch_size[0], y2 + patch_size[1]
    cv2.rectangle(img, (x2, y2), (x2_end, y2_end), color_translated, thickness)
    cv2.putText(img, 'Translated', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_translated, 2)

    # Visualize the perturbed patch with 4 corners in purple, if needed
    if show_perturbed:
        perturbed_points = np.array(perturbed_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [perturbed_points], isClosed=True, color=color_perturbed, thickness=thickness)
        cv2.putText(img, 'Perturbed', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_perturbed, 2)
        
    # Save the visualized image
    cv2.imwrite(save_path, img)
    print(f"Saved visualized patches to {save_path}")
    
# Function to generate initial random patch coordinates
def generate_patch(image, patch_size, center_bias_factor=0.2):
    h, w = image.shape[:2]
    max_x = w - patch_size[0]
    max_y = h - patch_size[1]
    
    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2
    
    # Standard deviation for the Gaussian distribution as a fraction of image dimensions
    std_x, std_y = int(w * center_bias_factor), int(h * center_bias_factor)
    
    # Generate x and y coordinates with a bias towards the center
    x = int(np.clip(np.random.normal(center_x, std_x), 0, max_x))
    y = int(np.clip(np.random.normal(center_y, std_y), 0, max_y))
    
    print(f"Selected initial patch at biased position ({x}, {y})")

    # Generate the four corner points of the rectangular patch
    four_points = [(x, y), (x + patch_size[0], y), (x + patch_size[0], y + patch_size[1]), (x, y + patch_size[1])]

    return (x, y), four_points
  
def perturb_corners(four_points, rho, image_shape):
    h, w = image_shape[:2]
    perturbed_points = []
    for point in four_points:
        while True:
            delta_x = random.randint(-rho, rho)
            delta_y = random.randint(-rho, rho)
            perturbed_x = point[0] + delta_x
            perturbed_y = point[1] + delta_y
            # Ensure the perturbed point is within image bounds
            if 0 <= perturbed_x < w and 0 <= perturbed_y < h:
                perturbed_points.append((perturbed_x, perturbed_y))
                break
    return perturbed_points
 
def slide_patch(image, initial_coords, patch_size, overlap):
    x, y = initial_coords
    h, w = image.shape[:2]

    directions = ["right", "left", "down", "up", "down-right", "down-left", "up-right", "up-left"]
    slide_successful = False

    while not slide_successful and directions:
        direction = random.choice(directions)
        slide_x, slide_y = 0, 0

        if direction == "right":
            slide_x = int(patch_size[0] * (1 - overlap))  # Only move in x
            new_x, new_y = x + slide_x, y
            if new_x + patch_size[0] <= w:
                slide_successful = True
        elif direction == "left":
            slide_x = int(patch_size[0] * (1 - overlap))  # Only move in x
            new_x, new_y = x - slide_x, y
            if new_x >= 0:
                slide_successful = True
        elif direction == "down":
            slide_y = int(patch_size[1] * (1 - overlap))  # Only move in y
            new_x, new_y = x, y + slide_y
            if new_y + patch_size[1] <= h:
                slide_successful = True
        elif direction == "up":
            slide_y = int(patch_size[1] * (1 - overlap))  # Only move in y
            new_x, new_y = x, y - slide_y
            if new_y >= 0:
                slide_successful = True
        elif direction == "down-right":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            new_x, new_y = x + slide_x, y + slide_y
            if new_x + patch_size[0] <= w and new_y + patch_size[1] <= h:
                slide_successful = True
        elif direction == "down-left":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            new_x, new_y = x - slide_x, y + slide_y
            if new_x >= 0 and new_y + patch_size[1] <= h:
                slide_successful = True
        elif direction == "up-right":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            new_x, new_y = x + slide_x, y - slide_y
            if new_x + patch_size[0] <= w and new_y >= 0:
                slide_successful = True
        elif direction == "up-left":
            slide_x = int(patch_size[0] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            slide_y = int(patch_size[1] * (1 - overlap) / math.sqrt(2))  # Adjust for diagonal
            new_x, new_y = x - slide_x, y - slide_y
            if new_x >= 0 and new_y >= 0:
                slide_successful = True

        if not slide_successful:
            directions.remove(direction)
            print(f"Cannot slide {direction}, trying another direction.")

    if not slide_successful:
        raise ValueError("Unable to slide in any direction without exceeding image boundaries.")

    print(f"Sliding bounding box to new position ({new_x}, {new_y}) in direction '{direction}' with {overlap * 100}% overlap")

    # Generate the new four corner points for the slid patch
    new_four_points = [(new_x, new_y), (new_x + patch_size[0], new_y),
                       (new_x + patch_size[0], new_y + patch_size[1]), (new_x, new_y + patch_size[1])]

    return (new_x, new_y), new_four_points

# Function to compute the corners of an image after applying a homography
def compute_warped_corners(homography, original_corners):
    warped_corners = cv2.perspectiveTransform(np.float32([original_corners]), homography)
    return warped_corners[0].tolist()

# Function to clamp warped corners to valid image coordinates
def clamp_warped_corners(warped_corners, image_shape):
    h, w = image_shape[:2]
    clamped_corners = []
    for x, y in warped_corners:
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))
        clamped_corners.append((clamped_x, clamped_y))
    return clamped_corners

# Function to check if a patch contains black padding
def patch_contains_padding(image, patch_corners):
    # Convert patch corners to integer values
    patch_corners = np.array(patch_corners, np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [patch_corners], 255)

    # Extract the patch region from the image using the mask
    patch = cv2.bitwise_and(image, image, mask=mask)

    # Check if the patch contains any black (padded) areas
    return np.any(patch == 0)

def transform_patch_corners(homography, four_points):
    # Convert the corners to homogeneous coordinates
    points = np.array(four_points, dtype='float32')
    points = np.column_stack((points, np.ones((4, 1))))  # Convert to homogeneous coordinates (x, y, 1)
    
    # Apply the homography matrix
    transformed_points = np.dot(homography, points.T).T
    
    # Convert back to (x, y) coordinates from homogeneous coordinates
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)  # Divide by the third coordinate (z)
    transformed_points = transformed_points[:, :2]  # Take only the x and y values

    return transformed_points.tolist()

def is_patch_in_valid_region(warped_image, patch_corners):
    h, w = warped_image.shape[:2]
    for (x, y) in patch_corners:
        if x < 0 or x >= w or y < 0 or y >= h or warped_image[int(y), int(x)].all() == 0:
            # If the point is outside bounds or in a black area
            return False
    return True

# Function to adjust the patch to avoid black padding
def adjust_patch_if_padding(image, patch_corners):
    if patch_contains_padding(image, patch_corners):
        print("Patch contains black padding, adjusting patch bounds.")
        # Recompute or adjust the patch coordinates
        patch_corners = clamp_warped_corners(patch_corners, image.shape)
    return patch_corners

# Function to crop patch from image based on patch corners
def crop_patch_from_image(image, patch_corners, save_path):
    x_min = int(min([p[0] for p in patch_corners]))
    y_min = int(min([p[1] for p in patch_corners]))
    x_max = int(max([p[0] for p in patch_corners]))
    y_max = int(max([p[1] for p in patch_corners]))
    cropped_patch = image[y_min:y_max, x_min:x_max]
    cv2.imwrite(save_path, cropped_patch)
    print(f"Saved cropped patch to {save_path}")
    return cropped_patch


# Function to save metadata
def save_metadata(sample_dir, image_path, patch_size, overlap, rho, four_points, perturbed_four_points, new_four_points):
    metadata_path = os.path.join(sample_dir, "metadata.txt")
    with open(metadata_path, "w") as file:
        file.write(f"Image Path: {os.path.abspath(image_path)}\n")
        file.write(f"Patch Size: {patch_size}\n")
        file.write(f"Overlap: {overlap:.2f}\n")
        file.write(f"Rho: {rho}\n")
        file.write(f"Corners of Initial Patch: {four_points}\n")
        file.write(f"Corners of Perturbed Polygon: {perturbed_four_points}\n")
        file.write(f"Corners of Translated Patch: {new_four_points}\n")
    print(f"Metadata saved to {metadata_path}")


# Parameters
patch_size = (256, 256)  # Size of the patch

def generate_samples(num_samples, image_path, output_dir, patch_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path or integrity.")
        return

    h, w = image.shape[:2]
    original_image_corners = [(0, 0), (w, 0), (w, h), (0, h)]

    for i in range(num_samples):
        overlap = random.uniform(0.25, 0.55)
        rho = random.randint(15, 50)
        
        initial_coords, four_points = generate_patch(image, patch_size, center_bias_factor=0.2)
        perturbed_four_points = perturb_corners(four_points, rho, image.shape)
        new_coords, new_four_points = slide_patch(image, initial_coords, patch_size, overlap)

        original_points = np.float32(four_points)
        perturbed_points = np.float32(perturbed_four_points)
        H = cv2.getPerspectiveTransform(original_points, perturbed_points)
        
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Error: Homography matrix is singular and cannot be inverted.")
            continue

        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))

        warped_image = cv2.warpPerspective(image, H_inv, (w, h))

        # Transform the patch coordinates with the homography to get their positions in the warped image
        transformed_patch_corners = transform_patch_corners(H_inv, new_four_points)

        # Check if the translated patch is within the valid region of the warped image
        if not is_patch_in_valid_region(warped_image, transformed_patch_corners):
            print(f"Sample {i+1}: Translated patch overlaps with black padding, skipping this sample.")
            continue

        warped_image_corners = compute_warped_corners(H_inv, original_image_corners)
        warped_image_corners = clamp_warped_corners(warped_image_corners, image.shape)

        initial_in_bounds = patch_contains_padding(warped_image, four_points)
        translated_in_bounds = patch_contains_padding(warped_image, new_four_points)

        sample_dir = os.path.join(output_dir, f'sample_{i+1}')
        os.makedirs(sample_dir, exist_ok=True)

        # 1. Save the full input image with the original and translated bounding boxes and perturbed points (in purple)
        visualize_patches(image.copy(), initial_coords, new_coords, perturbed_four_points, patch_size, save_path=os.path.join(sample_dir, 'input_image_with_boxes.png'), show_perturbed=True)
        
        # 2. Save the full warped image with the original and translated bounding boxes only (no perturbed points)
        visualize_patches(warped_image.copy(), initial_coords, new_coords, perturbed_four_points, patch_size, save_path=os.path.join(sample_dir, 'warped_image_with_boxes.png'), show_perturbed=False)

        if initial_in_bounds:
            crop_patch_from_image(image, four_points, os.path.join(sample_dir, 'initial_patch.png'))
        
        if translated_in_bounds:
            adjusted_translated_patch = adjust_patch_if_padding(warped_image, new_four_points)
            crop_patch_from_image(warped_image, adjusted_translated_patch, os.path.join(sample_dir, 'translated_patch.png'))
        
        save_metadata(sample_dir, image_path, patch_size, overlap, rho, four_points, perturbed_four_points, new_four_points)
        np.savetxt(os.path.join(sample_dir, 'homography_matrix.txt'), H)
        np.savetxt(os.path.join(sample_dir, 'delta_h.txt'), H_four_points)

        print(f"Sample {i+1} generated and saved in {sample_dir}")
        
def populate_generated_dataset(colormap_transformed_dataset, raw_thermal_dataset, generated_dataset_path):
    # Traverse through the colormap folders
    for colormap_folder in os.listdir(colormap_transformed_dataset):
        colormap_name = colormap_folder.replace("Dataset June2024 ", "")
        colormap_generated_path = os.path.join(generated_dataset_path, colormap_name)

        colormap_source_path = os.path.join(colormap_transformed_dataset, colormap_folder)
        
        for root, dirs, files in os.walk(colormap_source_path):
            relative_path = os.path.relpath(root, colormap_source_path)
            target_path = os.path.join(colormap_generated_path, relative_path)
            
            for file in files:
                if file.lower().endswith('.png'):
                    image_path = os.path.join(root, file)
                    sample_output_dir = os.path.join(target_path, file.replace('.png', ''))
                    generate_samples(num_samples, image_path, sample_output_dir)

    # Traverse through the raw thermal dataset
    raw_thermal_generated_path = os.path.join(generated_dataset_path, "Raw Thermal")
    
    for root, dirs, files in os.walk(raw_thermal_dataset):
        relative_path = os.path.relpath(root, raw_thermal_dataset)
        target_path = os.path.join(raw_thermal_generated_path, relative_path)
        
        for file in files:
            if file.lower().endswith('.png'):
                image_path = os.path.join(root, file)
                sample_output_dir = os.path.join(target_path, file.replace('.png', ''))
                generate_samples(num_samples, image_path, sample_output_dir)

    print("Generated dataset has been populated successfully.")

# Parameters for generating multiple samples
num_samples = 20
colormap_transformed_dataset = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Dataset/Colormap Transformed Data'
raw_thermal_dataset = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Dataset/Dataset June2024 Thermal'
output_base_dir = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography'
generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'

# Generate samples
is_structure_valid = verify_directory_structure(colormap_transformed_dataset, raw_thermal_dataset)
print("Directory structure is valid:", is_structure_valid)

# Run this after verifying the directory structure
if is_structure_valid:
    print("Create empty folder for generated data: ")
    create_generated_dataset(colormap_transformed_dataset, raw_thermal_dataset, output_base_dir)
    
    print("Populate data ...")
    populate_generated_dataset(colormap_transformed_dataset, raw_thermal_dataset, generated_dataset_path)
    
    print("Find results")
    compute_metrics_for_distance_folders(generated_dataset_path)
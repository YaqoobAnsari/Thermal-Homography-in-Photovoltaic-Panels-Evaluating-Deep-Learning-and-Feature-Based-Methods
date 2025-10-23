import os
import numpy as np
from matplotlib import cm
from tifffile import imread, imsave

# List of selected colormaps for comprehensive evaluation
colormaps = [
    # Perceptually Uniform Sequential
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    
    # Sequential (Application-Driven)
    'hot', 'cool', 'copper', 'Greys', 'Blues', 'Oranges', 'Reds',
    
    # Diverging
    'coolwarm', 'seismic', 'RdBu', 'BrBG', 'PiYG',
    
    # Cyclic
    'twilight', 'twilight_shifted', 'hsv',
    
    # Qualitative and Miscellaneous
    'tab10', 'Set1', 'Set2', 'Pastel1', 'terrain', 'nipy_spectral', 'flag'
]

# Function to create colormap-applied datasets as sibling folders
def create_colormap_datasets(source_dir, base_output_dir):
    for colormap_name in colormaps:
        # Define the output directory name for each colormap
        colormap_dir = os.path.join(base_output_dir, f'Dataset June2024 {colormap_name.capitalize()}')
        os.makedirs(colormap_dir, exist_ok=True)
        print(f"Created colormap directory: {colormap_dir}")
        
        # Create directory structure for the colormap
        for root, dirs, files in os.walk(source_dir):
            for directory in dirs:
                source_path = os.path.join(root, directory)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(colormap_dir, relative_path)
                os.makedirs(target_path, exist_ok=True)
                print(f"Ensured directory exists: {target_path}")

        # Process each image and apply the colormap
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.tif'):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, source_dir)

                    # Read the thermal image
                    try:
                        print(f"Processing image: {image_path}")
                        thermal_image = imread(image_path)

                        # Check if the image is valid
                        if thermal_image is None or thermal_image.size == 0:
                            print(f"Warning: {image_path} could not be read or is empty.")
                            continue
                        else:
                            print(f"Image {image_path} read successfully with shape {thermal_image.shape}.")

                        # Normalize and apply the colormap
                        colormap = cm.get_cmap(colormap_name)
                        normalized_image = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min())
                        colored_image = (colormap(normalized_image)[:, :, :3] * 65535).astype(np.uint16)  # Convert to uint16 RGB format

                        # Define output path and save the image
                        output_path = os.path.join(colormap_dir, relative_path, file.replace('.tif', f'_{colormap_name}.png'))
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        imsave(output_path, colored_image)
                        print(f"Saved image with {colormap_name} colormap to {output_path}")

                    except Exception as e:
                        print(f"Error processing {image_path} with colormap {colormap_name}: {e}")

# Define paths
source_thermal_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset\Dataset June2024 Thermal'
base_output_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset'

# Run the colormap application process for the selected colormaps
create_colormap_datasets(source_thermal_dir, base_output_dir)

print("All colormap datasets created successfully!")

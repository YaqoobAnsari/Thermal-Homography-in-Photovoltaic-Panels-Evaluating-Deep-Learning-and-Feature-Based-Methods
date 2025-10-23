import os
import shutil
import numpy as np
from tifffile import imsave
import subprocess
import json
import io
from PIL import Image
from matplotlib import pyplot as plt

def raw_to_thermal(filename):
    exiftool_path = r'C:\Users\yansari\AppData\Local\anaconda3\envs\Python_3.11\Scripts\exiftool.exe'
    """
    Extracts the thermal image as a 2D numpy array with raw temperature values.
    """
    try:
        # Read image metadata needed for conversion
        meta_json = subprocess.check_output(
            [exiftool_path, filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j', '-Model']
        )
        meta = json.loads(meta_json.decode())[0]

        # Extract the embedded thermal image
        thermal_img_bytes = subprocess.check_output([exiftool_path, "-RawThermalImage", "-b", filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)
        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        print(f"Successfully extracted thermal data from {filename}")
        return thermal_np

    except Exception as e:
        print(f"Failed to extract thermal data from {filename}: {e}")
        raise

def raw_to_rgb(filename):
    exiftool_path = r'C:\Users\yansari\AppData\Local\anaconda3\envs\Python_3.11\Scripts\exiftool.exe'
    """
    Extracts the RGB image embedded in the source image.
    """
    try:
        # Extract the embedded RGB image
        rgb_img_bytes = subprocess.check_output([exiftool_path, "-EmbeddedImage", "-b", filename])
        rgb_img_stream = io.BytesIO(rgb_img_bytes)
        rgb_img = Image.open(rgb_img_stream)
        rgb_np = np.array(rgb_img)

        print(f"Successfully extracted RGB data from {filename}")
        return rgb_np

    except Exception as e:
        print(f"Failed to extract RGB data from {filename}: {e}")
        raise

# Function to create directories recursively while maintaining the same structure
def create_directory_structure(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding directories in the target path
        for directory in dirs:
            source_path = os.path.join(root, directory)
            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)
            os.makedirs(target_path, exist_ok=True)
            print(f"Created directory: {target_path}")

# Function to process images and create thermal and RGB directories
def process_dataset(source_dir, thermal_dir, rgb_dir):
    # Create directory structures for thermal and RGB
    create_directory_structure(source_dir, thermal_dir)
    create_directory_structure(source_dir, rgb_dir)

    thermal_image_count = 0  # Counter for the number of thermal images processed

    # Iterate through the files and process each image
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)

                # Generate thermal image
                try:
                    thermal_image = raw_to_thermal(image_path)
                    thermal_output_path = os.path.join(thermal_dir, relative_path, file)
                    thermal_output_path = os.path.splitext(thermal_output_path)[0] + '_thermal.tif'  # Ensure .tif extension
                    imsave(thermal_output_path, thermal_image, dtype='uint16')  # Save as high-end .tif
                    print(f"Saved thermal image to {thermal_output_path}")
                    thermal_image_count += 1  # Increment counter
                except Exception as e:
                    print(f"Error processing thermal image for {image_path}: {e}")

                # Generate RGB image
                try:
                    rgb_image = raw_to_rgb(image_path)
                    rgb_output_path = os.path.join(rgb_dir, relative_path, file)
                    rgb_output_path = os.path.splitext(rgb_output_path)[0] + '_rgb.png'  # Ensure .png extension
                    Image.fromarray(rgb_image).save(rgb_output_path, format='PNG')  # Save as .png
                    print(f"Saved RGB image to {rgb_output_path}")
                except Exception as e:
                    print(f"Error processing RGB image for {image_path}: {e}")

    # Print total number of thermal images processed
    print(f"Total number of thermal images extracted and saved: {thermal_image_count}")

# Define paths
source_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset\Dataset June2024'
thermal_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset\Dataset June2024 Thermal'
rgb_dir = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset\Dataset June2024 RGB'
 
#process_dataset(source_dir, thermal_dir, rgb_dir)
image_path = 'C:/Users/yansari/Desktop/Workstation/Thermal Homography/20241210_123846/20241210_123851_261_R.JPG'
thermal_image = raw_to_thermal(image_path)
thermal_output_path = 'C:/Users/yansari/Desktop/Workstation/Thermal Homography/20241210_123846/20241210_123851_261_R'
thermal_output_path = os.path.splitext(thermal_output_path)[0] + '_thermal.tif'  # Ensure .tif extension
imsave(thermal_output_path, thermal_image, dtype='uint16')

rgb_image = raw_to_rgb(image_path)
rgb_output_path = 'C:/Users/yansari/Desktop/Workstation/Thermal Homography/20241210_123846/20241210_123851_261_R'
rgb_output_path = os.path.splitext(rgb_output_path)[0] + '_rgb.png'  # Ensure .png extension
Image.fromarray(rgb_image).save(rgb_output_path, format='PNG') 

print("Processing complete!")

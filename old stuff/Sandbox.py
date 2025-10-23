# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:39:32 2024

@author: yansari
"""

"""
Created on Sun Nov 10 14:12:59 2024

@author: yansari
"""
 
import os
import subprocess
import json
import io
from PIL import Image
import numpy as np

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

        # Print detailed information about the thermal image
        print(f"Successfully extracted thermal data from {filename}")
        print(f"Thermal image shape: {thermal_np.shape}")
        print(f"Thermal image data type: {thermal_np.dtype}")
        print(f"Thermal image byte length: {thermal_np.nbytes} bytes")
        print(f"Thermal image dimensions: {thermal_np.ndim}")

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

        # Print detailed information about the RGB image
        print(f"Successfully extracted RGB data from {filename}")
        print(f"RGB image shape: {rgb_np.shape}")
        print(f"RGB image data type: {rgb_np.dtype}")
        print(f"RGB image byte length: {rgb_np.nbytes} bytes")
        print(f"RGB image dimensions: {rgb_np.ndim}")

        return rgb_np

    except Exception as e:
        print(f"Failed to extract RGB data from {filename}: {e}")
        raise

# Define the image path
image_path = r'C:\Users\yansari\Desktop\Workstation\Thermal Homography\Dataset\Dataset June2024\June09_12pm\30deg\35\20240609_101232_567_R.JPG'

# Extract and print information for the thermal image
thermal_image = raw_to_thermal(image_path)

# Extract and print information for the RGB image
rgb_image = raw_to_rgb(image_path)

 
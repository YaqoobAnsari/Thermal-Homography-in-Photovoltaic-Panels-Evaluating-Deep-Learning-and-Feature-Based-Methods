# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:52:14 2024

@author: yansari
"""

import ezdxf
import matplotlib.pyplot as plt
from PIL import Image
import os

def dxf_to_png(input_dxf, output_png, dpi=300):
    """
    Convert a DXF file to PNG with the specified DPI.
    
    :param input_dxf: Path to the input DXF file.
    :param output_png: Path to save the output PNG.
    :param dpi: Dots per inch for PNG resolution.
    """
    try:
        # Load the DXF file
        doc = ezdxf.readfile(input_dxf)
        msp = doc.modelspace()
        
        # Get the bounding box
        bbox = msp.bbox()
        if bbox is None:
            raise ValueError("DXF file does not have a valid bounding box.")
        
        # Plot using matplotlib
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        
        # Render DXF entities to matplotlib axes
        for entity in msp:
            if entity.dxftype() in ("LINE", "CIRCLE", "ARC", "POLYLINE", "LWPOLYLINE"):
                entity.render(ax)
        
        ax.set_xlim(bbox.extmin.x, bbox.extmax.x)
        ax.set_ylim(bbox.extmin.y, bbox.extmax.y)
        ax.axis("off")
        
        # Save the figure as a high-resolution image
        plt.savefig("temp.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        
        # Convert to high-quality PNG with Pillow
        with Image.open("temp.png") as img:
            img.save(output_png, dpi=(dpi, dpi))
        
        os.remove("temp.png")  # Clean up temporary file
        print(f"PNG saved at {output_png}")
    except Exception as e:
        print(f"Error: {e}")

# Usage example
dxf_file = "C:/Users/yansari/Desktop/Workstation/Thermal Homography/FF-Generic-Mar2019.dwg"  # Output from dwg2dxf
output_png = "FF-Generic-Mar2019.png"
dxf_to_png(dxf_file, output_png, dpi=600)  # Save at 600 DPI

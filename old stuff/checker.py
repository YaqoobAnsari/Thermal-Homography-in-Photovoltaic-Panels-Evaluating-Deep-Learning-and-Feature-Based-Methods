"""
Verify Preprocessed Images dataset integrity and visualize random sample
Checks all height folders have images, displays one random TIFF
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetChecker:
    """Verify preprocessed dataset completeness"""
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.stats = {
            'total_height_folders': 0,
            'empty_folders': [],
            'folder_counts': {},
            'all_images': []
        }
    
    def check_structure(self):
        """Check all height folders and count images"""
        logger.info("Checking dataset structure...")
        
        # Find all height folders (10cm, 20cm, 30cm, 40cm)
        height_pattern = ['10cm', '20cm', '30cm', '40cm']
        
        for day_folder in sorted(self.root_dir.iterdir()):
            if not day_folder.is_dir() or not day_folder.name.startswith('2024-12'):
                continue
            
            for time_folder in sorted(day_folder.iterdir()):
                if not time_folder.is_dir():
                    continue
                
                for height_folder in sorted(time_folder.iterdir()):
                    if not height_folder.is_dir():
                        continue
                    
                    # Check if it's a height folder
                    if height_folder.name not in height_pattern:
                        logger.warning(f"Unexpected folder: {height_folder}")
                        continue
                    
                    self.stats['total_height_folders'] += 1
                    
                    # Count images
                    images = list(height_folder.glob('*.tiff')) + list(height_folder.glob('*.tif'))
                    count = len(images)
                    
                    folder_path = f"{day_folder.name}/{time_folder.name}/{height_folder.name}"
                    self.stats['folder_counts'][folder_path] = count
                    
                    # Track for random selection
                    for img in images:
                        self.stats['all_images'].append(img)
                    
                    # Check if empty
                    if count == 0:
                        self.stats['empty_folders'].append(folder_path)
                        logger.error(f"EMPTY: {folder_path}")
                    else:
                        logger.debug(f"✓ {folder_path}: {count} images")
    
    def print_report(self):
        """Print verification report"""
        print("\n" + "="*70)
        print("DATASET VERIFICATION REPORT")
        print("="*70)
        
        print(f"\nDataset: {self.root_dir}")
        print(f"\nStatistics:")
        print(f"  Total height folders: {self.stats['total_height_folders']}")
        print(f"  Empty folders: {len(self.stats['empty_folders'])}")
        print(f"  Total images: {len(self.stats['all_images'])}")
        
        if self.stats['empty_folders']:
            print(f"\n❌ EMPTY FOLDERS FOUND ({len(self.stats['empty_folders'])}):")
            for folder in self.stats['empty_folders']:
                print(f"  - {folder}")
        else:
            print(f"\n✓ All height folders contain images!")
        
        # Show distribution
        print(f"\nImage distribution by day:")
        day_counts = {}
        for folder_path, count in self.stats['folder_counts'].items():
            day = folder_path.split('/')[0]
            day_counts[day] = day_counts.get(day, 0) + count
        
        for day in sorted(day_counts.keys()):
            print(f"  {day}: {day_counts[day]} images")
        
        print("\n" + "="*70 + "\n")
    
    def visualize_random_image(self):
        """Display a random TIFF image"""
        if not self.stats['all_images']:
            logger.error("No images found to visualize")
            return
        
        # Select random image
        random_img_path = random.choice(self.stats['all_images'])
        
        logger.info(f"Visualizing random image: {random_img_path.name}")
        logger.info(f"Location: {random_img_path.parent}")
        
        # Read image
        img = cv2.imread(str(random_img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logger.error(f"Failed to read image: {random_img_path}")
            return
        
        # Compute stats
        img_stats = {
            'shape': img.shape,
            'dtype': img.dtype,
            'min': img.min(),
            'max': img.max(),
            'mean': img.mean(),
            'std': img.std()
        }
        
        # Display
        plt.figure(figsize=(12, 8))
        
        # Main image
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Random Sample: {random_img_path.name}", fontsize=10)
        plt.axis('off')
        
        # Histogram
        plt.subplot(2, 2, 2)
        plt.hist(img.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        plt.title("Intensity Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # Stats text
        plt.subplot(2, 2, 3)
        plt.axis('off')
        stats_text = f"""
Image Statistics:
━━━━━━━━━━━━━━━━━━━━
Shape:  {img_stats['shape']}
Dtype:  {img_stats['dtype']}
Min:    {img_stats['min']}
Max:    {img_stats['max']}
Mean:   {img_stats['mean']:.2f}
Std:    {img_stats['std']:.2f}

Path:
{random_img_path.parent.name}/{random_img_path.name}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # Zoomed section (center crop)
        plt.subplot(2, 2, 4)
        h, w = img.shape
        crop_size = min(h, w) // 3
        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        cropped = img[y_start:y_start+crop_size, x_start:x_start+crop_size]
        plt.imshow(cropped, cmap='gray', vmin=0, vmax=255)
        plt.title("Center Crop (Zoomed)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nDisplayed: {random_img_path.relative_to(self.root_dir)}")
        print(f"Image info: {img_stats['shape']}, {img_stats['dtype']}, range=[{img_stats['min']}, {img_stats['max']}]")


def main():
    preprocessed_root = Path(r"D:\kuyfavsksuyvsakuvcsa\Preprocessed Images")
    
    if not preprocessed_root.exists():
        print(f"ERROR: {preprocessed_root} not found")
        return
    
    print("="*70)
    print("PREPROCESSED DATASET CHECKER")
    print("="*70)
    print(f"\nDataset: {preprocessed_root}")
    print("\nThis will:")
    print("  1. Check all height folders (10cm-40cm) have images")
    print("  2. Count images per folder")
    print("  3. Display one random TIFF image with statistics")
    
    # Run checker
    checker = DatasetChecker(preprocessed_root)
    checker.check_structure()
    checker.print_report()
    
    # Visualize
    if checker.stats['all_images']:
        print("\nDisplaying random image...")
        checker.visualize_random_image()
    else:
        print("\n❌ No images found in dataset!")
    
    # Final verdict
    if checker.stats['empty_folders']:
        print(f"\n⚠ WARNING: {len(checker.stats['empty_folders'])} empty folders found!")
        print("Dataset may be incomplete.")
    else:
        print("\n✓ SUCCESS: All folders contain images!")
        print("Dataset structure is valid.")


if __name__ == "__main__":
    main()
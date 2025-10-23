"""
Organize preprocessed (enhanced) images from Dataset into clean structure
Works with ANY folder structure - just finds 'enhanced' folders and copies them
"""

import shutil
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organize enhanced images into clean preprocessed structure"""
    
    def __init__(self, source_root: Path, target_root: Path):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.stats = {
            'total_enhanced_folders': 0,
            'empty_enhanced_folders': 0,
            'images_copied': 0,
            'errors': [],
            'processed_paths': []
        }
    
    def extract_height(self, folder_name: str) -> str:
        """Try to extract height from folder name like '20241221_075958_10cm' -> '10cm'"""
        match = re.search(r'_(\d+cm)$', folder_name)
        if match:
            return match.group(1)
        return None
    
    def find_all_enhanced_folders(self):
        """
        Recursively find ALL 'enhanced' folders anywhere in the Dataset tree
        Preserves complete directory structure
        """
        enhanced_folders = []
        
        logger.info("Scanning entire dataset tree for 'enhanced' folders...")
        
        for enhanced_path in self.source_root.rglob('enhanced'):
            if not enhanced_path.is_dir():
                continue
            
            # Get parent folder (the folder containing 'enhanced')
            parent_folder = enhanced_path.parent
            
            # Get the full relative path from Dataset root to parent
            try:
                rel_path_to_parent = parent_folder.relative_to(self.source_root)
            except ValueError:
                logger.error(f"Path not relative to source: {parent_folder}")
                continue
            
            # Convert path parts to list
            parts = list(rel_path_to_parent.parts)
            
            # Try to extract height from the last part (parent folder name)
            height = self.extract_height(parts[-1]) if parts else None
            
            # If we found a height pattern, replace the timestamped folder with just the height
            # Otherwise, keep the original folder name
            if height:
                parts[-1] = height
                logger.debug(f"Extracted height '{height}' from {parent_folder.name}")
            else:
                # Keep original name but log it
                logger.debug(f"No height pattern in '{parent_folder.name}', keeping original name")
            
            # Reconstruct target path
            target_rel_path = Path(*parts) if parts else Path()
            target_dir = self.target_root / target_rel_path
            
            # Count images
            images = list(enhanced_path.glob('*.tiff')) + list(enhanced_path.glob('*.tif'))
            
            if len(images) == 0:
                self.stats['empty_enhanced_folders'] += 1
                self.stats['errors'].append(f"Empty: {rel_path_to_parent}")
                logger.warning(f"Empty enhanced folder: {rel_path_to_parent}")
            else:
                self.stats['total_enhanced_folders'] += 1
                self.stats['processed_paths'].append(str(rel_path_to_parent))
                
                enhanced_folders.append({
                    'source': enhanced_path,
                    'target': target_dir,
                    'images': images,
                    'source_rel': str(rel_path_to_parent),
                    'target_rel': str(target_rel_path),
                    'has_height': height is not None
                })
        
        return enhanced_folders
    
    def copy_enhanced_images(self, enhanced_folders):
        """Copy images from enhanced folders to organized structure"""
        
        for i, folder_info in enumerate(enhanced_folders, 1):
            source_path = folder_info['source']
            target_dir = folder_info['target']
            images = folder_info['images']
            
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all images
            copied_count = 0
            for img in images:
                target_path = target_dir / img.name
                
                try:
                    shutil.copy2(img, target_path)
                    copied_count += 1
                    self.stats['images_copied'] += 1
                except Exception as e:
                    self.stats['errors'].append(f"Copy failed {img.name}: {e}")
                    logger.error(f"Failed to copy {img.name}: {e}")
            
            height_marker = "✓" if folder_info['has_height'] else "○"
            logger.info(f"[{i}/{len(enhanced_folders)}] {height_marker} {folder_info['source_rel']} → {folder_info['target_rel']} ({copied_count} images)")
    
    def print_report(self):
        """Print summary report"""
        print("\n" + "="*70)
        print("DATASET ORGANIZATION REPORT")
        print("="*70)
        
        print(f"\nSource: {self.source_root}")
        print(f"Target: {self.target_root}")
        
        print(f"\nStatistics:")
        print(f"  Enhanced folders processed: {self.stats['total_enhanced_folders']}")
        print(f"  Empty enhanced folders: {self.stats['empty_enhanced_folders']}")
        print(f"  Total images copied: {self.stats['images_copied']}")
        
        print(f"\nProcessed paths (showing first 50):")
        for path in sorted(self.stats['processed_paths'])[:50]:
            print(f"  ✓ {path}")
        if len(self.stats['processed_paths']) > 50:
            print(f"  ... and {len(self.stats['processed_paths']) - 50} more")
        
        if self.stats['errors'] and self.stats['empty_enhanced_folders'] > 0:
            print(f"\nEmpty folders ({self.stats['empty_enhanced_folders']}):")
            empty_errors = [e for e in self.stats['errors'] if e.startswith('Empty:')]
            for i, error in enumerate(empty_errors[:10], 1):
                print(f"  {i}. {error}")
            if len(empty_errors) > 10:
                print(f"  ... and {len(empty_errors) - 10} more")
        
        print("\n" + "="*70 + "\n")
    
    def organize(self):
        """Main organization process"""
        logger.info("="*70)
        logger.info("STARTING DATASET ORGANIZATION")
        logger.info("="*70)
        logger.info(f"Source: {self.source_root}")
        logger.info(f"Target: {self.target_root}\n")
        
        # Find all enhanced folders recursively
        enhanced_folders = self.find_all_enhanced_folders()
        
        logger.info(f"\nScan Results:")
        logger.info(f"  Enhanced folders found: {self.stats['total_enhanced_folders']}")
        logger.info(f"  Empty folders: {self.stats['empty_enhanced_folders']}")
        
        if enhanced_folders:
            total_images = sum(len(f['images']) for f in enhanced_folders)
            with_height = sum(1 for f in enhanced_folders if f['has_height'])
            logger.info(f"  Total images to copy: {total_images}")
            logger.info(f"  Folders with height pattern: {with_height}/{len(enhanced_folders)}")
        
        if not enhanced_folders:
            logger.error("\nNo enhanced folders with images found!")
            return self.stats
        
        # Copy images
        logger.info("\nCopying images...\n")
        self.copy_enhanced_images(enhanced_folders)
        
        # Print report
        self.print_report()
        
        return self.stats


def main():
    # Paths
    source_root = Path(r"D:\kuyfavsksuyvsakuvcsa\Dataset")
    target_root = Path(r"D:\kuyfavsksuyvsakuvcsa\Preprocessed Images")
    
    if not source_root.exists():
        print(f"ERROR: Source directory not found: {source_root}")
        return
    
    print("="*70)
    print("DATASET ORGANIZER - Universal Enhanced Folder Copier")
    print("="*70)
    print(f"\nSource: {source_root}")
    print(f"Target: {target_root}")
    print(f"\nThis script will:")
    print("  1. Find ALL 'enhanced' folders anywhere in Dataset tree")
    print("  2. Preserve complete directory structure (day/time/etc)")
    print("  3. Extract height from folder names when possible (e.g., _10cm)")
    print("  4. Keep original folder names when no height pattern found")
    print("  5. Copy all images to organized 'Preprocessed Images' directory")
    print("\nExamples:")
    print("  Dataset/2024-12-21/8/20241221_075958_10cm/enhanced/")
    print("  → Preprocessed Images/2024-12-21/8/10cm/")
    print()
    print("  Dataset/2024-12-21/10/20241221_100711/enhanced/")
    print("  → Preprocessed Images/2024-12-21/10/20241221_100711/")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Operation cancelled.")
        return
    
    # Run
    organizer = DatasetOrganizer(source_root, target_root)
    stats = organizer.organize()
    
    # Summary
    if stats['total_enhanced_folders'] > 0:
        print(f"✓ SUCCESS! Copied {stats['images_copied']} images from {stats['total_enhanced_folders']} enhanced folders.")
    else:
        print("No enhanced folders found to process.")


if __name__ == "__main__":
    main()
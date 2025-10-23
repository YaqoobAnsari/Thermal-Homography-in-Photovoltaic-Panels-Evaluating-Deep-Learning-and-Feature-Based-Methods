import cv2
import numpy as np 
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import euclidean
import os
import pandas as pd
import time

def compute_ace(homography, image_shape):
    if homography is None or not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        #print("Homography is invalid or not computed, skipping ACE calculation.")
        return float('inf')  # Return a large value to indicate the error

    h, w = image_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32').reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, homography)
    ace = np.mean([euclidean(corner[0], projected_corner[0]) for corner, projected_corner in zip(corners, projected_corners)])
    return ace


def compute_ssim(image_1, image_2):
    # Determine the smaller side of the images
    min_side = min(image_1.shape[:2])  # Get the smaller dimension
    win_size = min(7, min_side)  # Use 7 or the smallest dimension, whichever is smaller
    
    # Ensure the win_size is odd
    if win_size % 2 == 0:
        win_size -= 1
    
    # Compute SSIM with the adjusted window size
    return ssim(image_1, image_2, win_size=win_size)

def compute_mi(image_1, image_2):
    hist_2d, _, _ = np.histogram2d(image_1.ravel(), image_2.ravel(), bins=20)
    return mutual_info_score(None, None, contingency=hist_2d)

def compute_afrr(keypoints_1, keypoints_2, matches, homography, threshold=5.0):
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    projected_pts = cv2.perspectiveTransform(src_pts, homography)
    
    correct_matches = 0
    for projected, dst in zip(projected_pts, dst_pts):
        if euclidean(projected[0], dst[0]) < threshold:
            correct_matches += 1
            
    afrr = correct_matches / len(matches) if matches else 0
    return afrr
 
def match_and_compute_homography(detector, image_1, image_2, method_name, method='RANSAC'):
    #print(f"Running {method_name} ({method})...")
    
    # Detect keypoints and descriptors
    keypoints_1, descriptors_1 = detector.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(image_2, None)

    if descriptors_1 is None or descriptors_2 is None:
        #print(f"     Skipping {method_name} due to no keypoints found.")
        return None, None, None, None

    # Use BFMatcher for feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        # Ensure there are at least two matches to apply Lowe's ratio test
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        #print(f"     Not enough matches found for {method_name}. Skipping homography computation.")
        return None, keypoints_1, keypoints_2, good_matches

    # Extract the matched keypoints
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography using the specified method
    homography = None
    if method == 'RANSAC':
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    elif method == 'RANSAC++':
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000)
    elif method == 'MAGSAC':
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.USAC_MAGSAC)
    elif method == 'MAGSAC++':
        magsac_plus_threshold = 1.0
        max_iterations = 15000
        homography, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=magsac_plus_threshold, maxIters=max_iterations)
    
    if homography is None:
        #print(f"{method_name}: Homography could not be computed, skipping.")
        return None, keypoints_1, keypoints_2, good_matches

    return homography, keypoints_1, keypoints_2, good_matches
  


def evaluate_metrics(image_1, image_2, homography, keypoints_1, keypoints_2, matches, method_name):
    #print(f"Evaluating metrics for {method_name}...")

    if homography is None:
        #print(f"{method_name}: Homography is None, skipping metric evaluation.")
        return "F", "F", "F", "F"

    # Calculate each metric, handling None by returning "F" if any fail to compute
    ssim_value = compute_ssim(image_1, image_2) if image_1 is not None and image_2 is not None else "F"
    mi_value = compute_mi(image_1, image_2) if image_1 is not None and image_2 is not None else "F"
    afrr_value = compute_afrr(keypoints_1, keypoints_2, matches, homography) if matches else "F"
    ace_value = compute_ace(homography, image_1.shape) if homography is not None else "F"

    # Replace any None values with "F" if computation failed for any reason
    ssim_value = ssim_value if ssim_value is not None else "F"
    mi_value = mi_value if mi_value is not None else "F"
    afrr_value = afrr_value if afrr_value is not None else "F"
    ace_value = ace_value if ace_value is not None else "F"
    
    #print(f"{method_name}: ACE = {ace_value}, SSIM = {ssim_value}, MI = {mi_value}, AFRR = {afrr_value}")
    
    return ace_value, ssim_value, mi_value, afrr_value
def evaluate_methods(image_1, image_2):
    # Define improved detector parameters for each method
    methods = {
        "SIFT + RANSAC": (cv2.SIFT_create(
            nfeatures=10000, 
            contrastThreshold=0.002, 
            edgeThreshold=5, 
            sigma=2.6
        ), 'RANSAC'),
        
        "SIFT + RANSAC++": (cv2.SIFT_create(
            nfeatures=10000, 
            contrastThreshold=0.002, 
            edgeThreshold=5, 
            sigma=2.6
        ), 'RANSAC++'),
        
        "SIFT + MAGSAC": (cv2.SIFT_create(
            nfeatures=10000, 
            contrastThreshold=0.002, 
            edgeThreshold=5, 
            sigma=2.6
        ), 'MAGSAC'),
        
        "SIFT + MAGSAC++": (cv2.SIFT_create(
            nfeatures=10000, 
            contrastThreshold=0.002, 
            edgeThreshold=5, 
            sigma=2.6
        ), 'MAGSAC++'),
        
        "ORB + RANSAC": (cv2.ORB_create(
            nfeatures=12000, 
            scaleFactor=1.2, 
            nlevels=12, 
            edgeThreshold=10, 
            patchSize=70, 
            fastThreshold=15
        ), 'RANSAC'),
        
        "ORB + RANSAC++": (cv2.ORB_create(
            nfeatures=12000, 
            scaleFactor=1.2, 
            nlevels=12, 
            edgeThreshold=10, 
            patchSize=70, 
            fastThreshold=15
        ), 'RANSAC++'),
        
        "ORB + MAGSAC": (cv2.ORB_create(
            nfeatures=12000, 
            scaleFactor=1.2, 
            nlevels=12, 
            edgeThreshold=10, 
            patchSize=70, 
            fastThreshold=15
        ), 'MAGSAC'),
        
        "ORB + MAGSAC++": (cv2.ORB_create(
            nfeatures=12000, 
            scaleFactor=1.2, 
            nlevels=12, 
            edgeThreshold=10, 
            patchSize=70, 
            fastThreshold=15
        ), 'MAGSAC++'),
        
        "BRISK + RANSAC": (cv2.BRISK_create(
            thresh=25, 
            octaves=5, 
            patternScale=1.8
        ), 'RANSAC'),
        
        "BRISK + RANSAC++": (cv2.BRISK_create(
            thresh=25, 
            octaves=5, 
            patternScale=1.8
        ), 'RANSAC++'),
        
        "BRISK + MAGSAC": (cv2.BRISK_create(
            thresh=25, 
            octaves=5, 
            patternScale=1.8
        ), 'MAGSAC'),
        
        "BRISK + MAGSAC++": (cv2.BRISK_create(
            thresh=25, 
            octaves=5, 
            patternScale=1.8
        ), 'MAGSAC++'),
        
        "AKAZE + RANSAC": (cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
            descriptor_size=0, 
            descriptor_channels=3, 
            threshold=0.0012, 
            nOctaves=5, 
            nOctaveLayers=4
        ), 'RANSAC'),
        
        "AKAZE + RANSAC++": (cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
            descriptor_size=0, 
            descriptor_channels=3, 
            threshold=0.0012, 
            nOctaves=5, 
            nOctaveLayers=4
        ), 'RANSAC++'),
        
        "AKAZE + MAGSAC": (cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
            descriptor_size=0, 
            descriptor_channels=3, 
            threshold=0.0012, 
            nOctaves=5, 
            nOctaveLayers=4
        ), 'MAGSAC'),
        
        "AKAZE + MAGSAC++": (cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
            descriptor_size=0, 
            descriptor_channels=3, 
            threshold=0.0012, 
            nOctaves=5, 
            nOctaveLayers=4
        ), 'MAGSAC++'),
        
        "KAZE + RANSAC": (cv2.KAZE_create(
            extended=False, 
            upright=False, 
            threshold=0.0018, 
            nOctaves=6, 
            nOctaveLayers=5
        ), 'RANSAC'),
        
        "KAZE + RANSAC++": (cv2.KAZE_create(
            extended=False, 
            upright=False, 
            threshold=0.0018, 
            nOctaves=6, 
            nOctaveLayers=5
        ), 'RANSAC++'),
        
        "KAZE + MAGSAC": (cv2.KAZE_create(
            extended=False, 
            upright=False, 
            threshold=0.0018, 
            nOctaves=6, 
            nOctaveLayers=5
        ), 'MAGSAC'),
        
        "KAZE + MAGSAC++": (cv2.KAZE_create(
            extended=False, 
            upright=False, 
            threshold=0.0018, 
            nOctaves=6, 
            nOctaveLayers=5
        ), 'MAGSAC++'),
    }

    results = {}
    for method_name, (detector, method) in methods.items():
        # Compute homography and evaluate metrics if possible
        homography, keypoints_1, keypoints_2, matches = match_and_compute_homography(detector, image_1, image_2, method_name, method)
        if homography is None or matches is None:
            # Add an empty entry for methods that failed
            results[method_name] = ("F", "F", "F", "F")
            continue
        
        # Compute metrics and handle the case where any metric returns None
        ace_value, ssim_value, mi_value, afrr_value = evaluate_metrics(image_1, image_2, homography, keypoints_1, keypoints_2, matches, method_name)
        results[method_name] = (
            ace_value if ace_value is not None else "F",
            ssim_value if ssim_value is not None else "F",
            mi_value if mi_value is not None else "F",
            afrr_value if afrr_value is not None else "F"
        )
    
    return results


def process_distance_folders(parent_folder):
    # Process each distance folder
    for distance_folder in os.listdir(parent_folder):
        distance_path = os.path.join(parent_folder, distance_folder)
        if not os.path.isdir(distance_path):
            continue
        
        print(f"      Processing Distance Folder: {distance_folder}")
        
        # Create the Results CSV folder
        results_csv_path = os.path.join(distance_path, "Results CSV")
        os.makedirs(results_csv_path, exist_ok=True)
        
        # Initialize method results
        method_results = {
            "SIFT + RANSAC": [],
            "SIFT + RANSAC++": [],
            "SIFT + MAGSAC": [],
            "SIFT + MAGSAC++": [],
            "ORB + RANSAC": [],
            "ORB + RANSAC++": [],
            "ORB + MAGSAC": [],
            "ORB + MAGSAC++": [],
            "BRISK + RANSAC": [],
            "BRISK + RANSAC++": [],
            "BRISK + MAGSAC": [],
            "BRISK + MAGSAC++": [],
            "AKAZE + RANSAC": [],
            "AKAZE + RANSAC++": [],
            "AKAZE + MAGSAC": [],
            "AKAZE + MAGSAC++": [],
            "KAZE + RANSAC": [],
            "KAZE + RANSAC++": [],
            "KAZE + MAGSAC": [],
            "KAZE + MAGSAC++": []
        }
        
        # Traverse each image folder inside the distance folder
        for image_folder in os.listdir(distance_path):
            image_folder_path = os.path.join(distance_path, image_folder)
            if not os.path.isdir(image_folder_path):
                continue
            
            # Traverse each sample in the image folder
            for sample_folder in os.listdir(image_folder_path):
                sample_folder_path = os.path.join(image_folder_path, sample_folder)
                if not os.path.isdir(sample_folder_path):
                    continue
                
                # Load the images
                initial_image_path = os.path.join(sample_folder_path, "initial_patch.png")
                translated_image_path = os.path.join(sample_folder_path, "translated_patch.png")
                if not (os.path.exists(initial_image_path) and os.path.exists(translated_image_path)):
                    print(f"        Skipping {sample_folder}: Missing required images.")
                    continue
                 
                image_1 = cv2.imread(initial_image_path, cv2.IMREAD_GRAYSCALE)
                image_2 = cv2.imread(translated_image_path, cv2.IMREAD_GRAYSCALE)
                
                if image_1 is None or image_2 is None:
                    print(f"        Skipping {sample_folder}: Could not read images.")
                    continue
                
                # Evaluate all methods
                method_metrics = evaluate_methods(image_1, image_2)
                for method_name, metrics in method_metrics.items():
                    ace, ssim_val, mi, afrr = metrics
                    method_results[method_name].append({
                        "File Name": f"{image_folder}/{sample_folder}",
                        "ACE": ace,
                        "SSIM": ssim_val,
                        "MI": mi,
                        "AFRR": afrr
                    })
        
        # Save results for each method in a separate CSV
        for method_name, rows in method_results.items():
            csv_file_path = os.path.join(results_csv_path, f"{method_name}_metrics.csv")
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_file_path, index=False)
                print(f"        Results for {method_name}, saved!")
 
def compute_metrics_for_distance_folders(generated_dataset_path):
    """
    Traverse the dataset directory and compute metrics for each folder.
    Adds timing to measure the processing time of each timing folder.
    """
    # Traverse each testing folder (26 or so)
    for testing_folder in os.listdir(generated_dataset_path):
        testing_path = os.path.join(generated_dataset_path, testing_folder)
        if not os.path.isdir(testing_path):
            continue
        
        print(f"Processing Testing Folder: {testing_folder}")
        
        # Traverse each timing folder (June02_12pm, June05_8am, June09_12pm)
        for timing_folder in os.listdir(testing_path):
            timing_path = os.path.join(testing_path, timing_folder)
            if not os.path.isdir(timing_path):
                continue

            print(f"  Processing Timing Folder: {timing_folder}")
            
            # Start timer for this timing folder
            start_time = time.time()
            
            if timing_folder == "June02_12pm":
                # Directly process Distance Folders for June02_12pm
                process_distance_folders(timing_path)
            else:
                # Process Degree Folders for other timing folders
                for degree_folder in os.listdir(timing_path):
                    degree_path = os.path.join(timing_path, degree_folder)
                    if not os.path.isdir(degree_path):
                        continue
                    
                    print(f"    Processing Degree Folder: {degree_folder}")
                    process_distance_folders(degree_path)

            # End timer and print the elapsed time
            elapsed_time = time.time() - start_time
            print(f"  Completed Timing Folder: {timing_folder} in {elapsed_time:.2f} seconds")

def validate_and_fix_images(base_path):
    """
    Traverse the directory structure, validate image pairs, resize mismatched images,
    and report any issues concisely.
    """
    total_colormaps = 0
    total_issues = 0
    total_resizes = 0

    print("Starting validation and fixing of dataset...\n")

    # Traverse colormap folders
    for colormap_folder in os.listdir(base_path):
        colormap_path = os.path.join(base_path, colormap_folder)
        if not os.path.isdir(colormap_path):
            continue

        print(f"Processing Colormap Folder: {colormap_folder}")

        # Traverse timing folders
        for timing_folder in os.listdir(colormap_path):
            timing_path = os.path.join(colormap_path, timing_folder)
            if not os.path.isdir(timing_path):
                continue

            print(f"  Timing Folder: {timing_folder}")
            all_good = True  # Track if the timing folder is valid

            # Check if the timing folder is June02_12pm (directly has distance folders)
            if timing_folder == "June02_12pm":
                distance_folders = [os.path.join(timing_path, d) for d in os.listdir(timing_path) if os.path.isdir(os.path.join(timing_path, d))]
            else:
                # Traverse degree folders and collect all distance folders
                degree_folders = [os.path.join(timing_path, d) for d in os.listdir(timing_path) if os.path.isdir(os.path.join(timing_path, d))]
                distance_folders = []
                for degree_folder in degree_folders:
                    distance_folders.extend(
                        [os.path.join(degree_folder, d) for d in os.listdir(degree_folder) if os.path.isdir(os.path.join(degree_folder, d))]
                    )

            # Process distance folders
            for distance_folder in distance_folders:
                for image_folder in os.listdir(distance_folder):
                    image_folder_path = os.path.join(distance_folder, image_folder)
                    if not os.path.isdir(image_folder_path):
                        continue

                    for sample_folder in os.listdir(image_folder_path):
                        sample_folder_path = os.path.join(image_folder_path, sample_folder)
                        if not os.path.isdir(sample_folder_path):
                            continue

                        # Paths for the two images
                        initial_image_path = os.path.join(sample_folder_path, "initial_patch.png")
                        translated_image_path = os.path.join(sample_folder_path, "translated_patch.png")

                        # Check if both images exist
                        if not os.path.exists(initial_image_path) or not os.path.exists(translated_image_path):
                            print(f"    [Issue] Missing images in {sample_folder_path}")
                            total_issues += 1
                            all_good = False
                            continue

                        # Read images
                        image_1 = cv2.imread(initial_image_path, cv2.IMREAD_GRAYSCALE)
                        image_2 = cv2.imread(translated_image_path, cv2.IMREAD_GRAYSCALE)

                        # Check if images are readable
                        if image_1 is None or image_2 is None:
                            print(f"    [Issue] Could not read images in {sample_folder_path}")
                            total_issues += 1
                            all_good = False
                            continue

                        # Check if dimensions match
                        if image_1.shape != image_2.shape:
                            print(f"    [Fixing] Dimension mismatch in {sample_folder_path}")
                            print(f"      Initial: {image_1.shape}, Translated: {image_2.shape}")

                            # Resize translated image to match initial patch dimensions
                            resized_image = cv2.resize(image_2, (image_1.shape[1], image_1.shape[0]))
                            cv2.imwrite(translated_image_path, resized_image)
                            print(f"      [Fixed] Resized translated_patch.png to match initial_patch.png")
                            total_resizes += 1

                            # Re-check the dimensions
                            resized_image_2 = cv2.imread(translated_image_path, cv2.IMREAD_GRAYSCALE)
                            if resized_image_2.shape != image_1.shape:
                                print(f"    [Error] Resizing failed in {sample_folder_path}")
                                total_issues += 1
                                all_good = False
                                continue

            # Print status for the timing folder
            if all_good:
                print(f"  [All Good] Timing Folder: {timing_folder}")

        total_colormaps += 1

    # Final summary
    print("\nValidation and Fixing Complete!")
    print(f"Total Colormaps Processed: {total_colormaps}")
    print(f"Total Issues Found: {total_issues}")
    print(f"Total Images Resized: {total_resizes}")

# Usage Example
generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'
#validate_and_fix_images(generated_dataset_path)
compute_metrics_for_distance_folders(generated_dataset_path) 


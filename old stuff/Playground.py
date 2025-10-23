import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, metrics
from skimage.restoration import denoise_bilateral
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image
import cv2

def detect_and_suppress_glare(thermal_array):
    """Detects and suppresses glare adaptively using statistical methods."""
    thermal_normalized = (thermal_array - np.min(thermal_array)) / (np.max(thermal_array) - np.min(thermal_array))

    # Compute mean and standard deviation of intensities
    mean_intensity = np.mean(thermal_normalized)
    std_intensity = np.std(thermal_normalized)

    # Define an adaptive glare threshold
    glare_threshold = mean_intensity + 2 * std_intensity

    # Generate glare mask
    glare_mask = thermal_normalized > glare_threshold

    # Adaptive suppression: Clamp glare pixels to the threshold
    thermal_corrected = thermal_normalized.copy()
    thermal_corrected[glare_mask] = glare_threshold

    # Normalize again after glare correction
    thermal_corrected = (thermal_corrected - np.min(thermal_corrected)) / (np.max(thermal_corrected) - np.min(thermal_corrected))

    return thermal_corrected

def preprocess_thermal_image(tiff_path, output_folder, save_intermediate=True, use_colormap=False):
    """
    Preprocess a thermal image with integrated glare suppression and save outputs.
    """
    # Load the thermal image
    thermal_image = Image.open(tiff_path)
    thermal_array = np.array(thermal_image)

    # **NEW: Save Raw Thermal Image**
    raw_thermal_normalized = (thermal_array - np.min(thermal_array)) / (np.max(thermal_array) - np.min(thermal_array))

    # Function to save an image with optional colormap
    def save_image(image, name_suffix):
        filename = os.path.splitext(os.path.basename(tiff_path))[0] + name_suffix + ".png"
        filepath = os.path.join(output_folder, filename)
        cmap = 'inferno' if use_colormap else 'gray'
        plt.imsave(filepath, image, cmap=cmap)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # **Save Raw Image Before Any Processing**
    save_image(raw_thermal_normalized, "_raw")

    # **Apply Glare Suppression**
    thermal_corrected = detect_and_suppress_glare(thermal_array)
    save_image(thermal_corrected, "_glare_suppressed")

    # Step 1: Apply CLAHE
    thermal_clahe = exposure.equalize_adapthist(thermal_corrected, clip_limit=0.018)
    if save_intermediate:
        save_image(thermal_clahe, "_clahe")

    # Step 2: Apply Bilateral Filtering
    thermal_denoised = denoise_bilateral(thermal_clahe, sigma_color=0.015, sigma_spatial=8)
    if save_intermediate:
        save_image(thermal_denoised, "_denoising")

    # Step 3: Sharpening
    thermal_sharpened = thermal_denoised + 0.5 * (thermal_denoised - gaussian_filter(thermal_denoised, sigma=1))
    save_image(thermal_sharpened, "_final_preprocessed")

    # Feature Extraction with ORB & SIFT
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()

    def compute_keypoints(image, name_suffix):
        """Detect and save keypoints using SIFT and ORB."""
        image_scaled = (image * 255).astype(np.uint8)
        sift_kp, sift_desc = sift.detectAndCompute(image_scaled, None)
        orb_kp, orb_desc = orb.detectAndCompute(image_scaled, None)
        return sift_kp, sift_desc, orb_kp, orb_desc

    # Extract keypoints from raw and processed images
    raw_sift_kp, raw_sift_desc, raw_orb_kp, raw_orb_desc = compute_keypoints(raw_thermal_normalized, "_raw")
    processed_sift_kp, processed_sift_desc, processed_orb_kp, processed_orb_desc = compute_keypoints(thermal_sharpened, "_final_preprocessed")

    # Compute inlier ratio for feature matching
    def match_descriptors(desc1, desc2, method="SIFT"):
        if desc1 is None or desc2 is None:
            return 0
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)) if method == "SIFT" else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.knnMatch(desc1, desc2, k=2) if method == "SIFT" else matcher.match(desc1, desc2)
        good_matches = [m for m, n in matches if m.distance < 0.72 * n.distance] if method == "SIFT" else matches
        return len(good_matches) / len(matches) if matches else 0

    # Compute metrics
    sift_inlier_ratio = match_descriptors(raw_sift_desc, processed_sift_desc, method="SIFT")
    orb_inlier_ratio = match_descriptors(raw_orb_desc, processed_orb_desc, method="ORB")

    # Compute entropy
    def compute_entropy(descriptors):
        if descriptors is None or len(descriptors) == 0:
            return 0
        hist, _ = np.histogram(descriptors, bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    raw_entropy = compute_entropy(raw_sift_desc) + compute_entropy(raw_orb_desc)
    processed_entropy = compute_entropy(processed_sift_desc) + compute_entropy(processed_orb_desc)

    # Compute Gradient Magnitude Similarity Deviation (GMSD)
    raw_gradient = sobel(raw_thermal_normalized)
    processed_gradient = sobel(thermal_sharpened)
    gmsd = np.sqrt(np.mean((raw_gradient - processed_gradient) ** 2))

    # Compute PSNR and SSIM
    psnr_final = metrics.peak_signal_noise_ratio(thermal_sharpened, raw_thermal_normalized, data_range=1)
    ssim_final = metrics.structural_similarity(thermal_sharpened, raw_thermal_normalized, data_range=1)

    return {
        "PSNR": psnr_final,
        "SSIM": ssim_final,
        "Number of Keypoints (Raw SIFT)": len(raw_sift_kp),
        "Number of Keypoints (Processed SIFT)": len(processed_sift_kp),
        "Number of Keypoints (Raw ORB)": len(raw_orb_kp),
        "Number of Keypoints (Processed ORB)": len(processed_orb_kp),
        "Inlier Ratio (SIFT)": sift_inlier_ratio,
        "Inlier Ratio (ORB)": orb_inlier_ratio,
        "Entropy (Raw)": raw_entropy,
        "Entropy (Processed)": processed_entropy,
        "GMSD": gmsd
    }

# Example Usage
if __name__ == "__main__":
    tiff_path = "20241210_123846/20241224_135940_858_IR.TIFF"
    output_folder = "Playground"
    
    metrics_result = preprocess_thermal_image(tiff_path, output_folder, save_intermediate=True, use_colormap=False)

    print("Evaluation Metrics:")
    for key, value in metrics_result.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

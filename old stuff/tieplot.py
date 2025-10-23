import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to randomly crop a patch
def random_crop(image, crop_size):
    h, w = image.shape[:2]
    top = np.random.randint(0, h - crop_size)
    left = np.random.randint(0, w - crop_size)
    patch = image[top:top + crop_size, left:left + crop_size]
    return patch, (top, left)

# Function to randomly perturb the corners
def random_perturb_corners(corners, perturb_range):
    perturbed_corners = []
    for corner in corners:
        perturb_x = np.random.randint(-perturb_range, perturb_range)
        perturb_y = np.random.randint(-perturb_range, perturb_range)
        perturbed_corners.append((corner[0] + perturb_x, corner[1] + perturb_y))
    return np.array(perturbed_corners, dtype=np.float32)

# Function to compute homography
def compute_homography(corners_A, corners_B):
    return cv2.getPerspectiveTransform(corners_A, corners_B)

# Function to warp an image using a homography matrix
def warp_image(image, H, crop_size, position):
    top, left = position
    patch = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    warped_patch = patch[top:top + crop_size, left:left + crop_size]
    return warped_patch

# Function to generate patches
def generate_patches(image, crop_size=256, perturb_range=50):
    patch_A, (top, left) = random_crop(image, crop_size)
    corners_A = np.array([
        [left, top], 
        [left + crop_size, top], 
        [left + crop_size, top + crop_size], 
        [left, top + crop_size]
    ], dtype=np.float32)
    corners_B = random_perturb_corners(corners_A, perturb_range)
    H_AB = compute_homography(corners_A, corners_B)
    patch_B = warp_image(image, H_AB, crop_size, (top, left))
    return patch_A, patch_B, corners_A, corners_B, H_AB

# Function to match features with ORB and create tie plots
def orb_and_plot_matches(patch_A, patch_B):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(patch_A, None)
    kp2, des2 = orb.detectAndCompute(patch_B, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = mask.ravel().tolist()
        inlier_score = sum(inliers) / len(inliers) * 100
    else:
        H = None
        inliers = []
        inlier_score = 0

    match_img = cv2.drawMatches(
        patch_A, kp1, patch_B, kp2, matches, None, 
        matchesMask=inliers if H is not None else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_img, inlier_score

# Load the thermal image
image_path = '/mnt/data/20241222_082037_657_IR_final_preprocessed.png'
thermal_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Generate two random patch pairs with sizable perturbation
patch_A1, patch_B1, _, _, _ = generate_patches(thermal_image, crop_size=256, perturb_range=50)
patch_A2, patch_B2, _, _, _ = generate_patches(thermal_image, crop_size=256, perturb_range=50)

# Perform ORB matching for both patch pairs
match_img1, inlier_score1 = orb_and_plot_matches(patch_A1, patch_B1)
match_img2, inlier_score2 = orb_and_plot_matches(patch_A2, patch_B2)

# Plot the results side by side with academic-style formatting
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(match_img1, cmap='gray')
axes[0].set_title(f"**Example 1: Inlier Score = {inlier_score1:.2f}%**", fontsize=15, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(match_img2, cmap='gray')
axes[1].set_title(f"**Example 2: Inlier Score = {inlier_score2:.2f}%**", fontsize=15, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

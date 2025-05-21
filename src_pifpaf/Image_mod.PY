import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import gc

print(">>> Running:", __file__)

# # Set random seeds for reproducibility
# np.random.seed(42)

# -----------------------------
# Augmentation Definition
# -----------------------------
def apply_fixed_color_correction(image: np.ndarray) -> np.ndarray:
    # ---- USER-EDITABLE PARAMETERS ----
    brightness_factor = 1.2  # Overall brightness multiplier
    contrast_factor = 1.0    # Contrast multiplier (1.0 = no change)
    r_shift = 0              # Add this value to RED channel
    g_shift = 0              # Add this value to GREEN channel
    b_shift = -100              # Add this value to BLUE channel
    # ----------------------------------

    img = np.array(image, copy=True).astype(np.float32)
    print("Input min/max:", img.min(), img.max())
    print("!!! DEBUG: This is the hardcoded brightness factor being used:", brightness_factor)
    print("!!! DEBUG: Contrast factor:", contrast_factor)
    print("!!! DEBUG: RGB shifts: R={}, G={}, B={}".format(r_shift, g_shift, b_shift))

    # Apply RGB tone correction
    img[..., 0] = np.clip(img[..., 0] + r_shift, 0, 255)  # Red
    img[..., 1] = np.clip(img[..., 1] + g_shift, 0, 255)  # Green
    img[..., 2] = np.clip(img[..., 2] + b_shift, 0, 255)  # Blue

    # Apply brightness
    img = img * brightness_factor

    # Apply contrast (centered at 128)
    img = (img - 128) * contrast_factor + 128

    print("After scaling min/max:", img.min(), img.max())
    img = np.clip(img, 0, 255)
    print("After clipping min/max:", img.min(), img.max())
    return img.astype(np.uint8)

color_aug = apply_fixed_color_correction

# -----------------------------
# Helper: Apply augmentation
# -----------------------------
def apply_augmentation(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    print("\nInput stats:")
    print("Shape:", image.shape)
    print("RGB means:", [image[...,i].mean() for i in range(3)])
    aug_img = color_aug(image)
    print("\nOutput stats:")
    print("Shape:", aug_img.shape)
    print("RGB means:", [aug_img[...,i].mean() for i in range(3)])
    # Show a few pixel values for debugging
    print("Sample original pixel:", image[0,0])
    print("Sample augmented pixel:", aug_img[0,0])
    tensor_result = ToTensorV2()(image=aug_img)['image']
    return tensor_result

# -----------------------------
# Mode: Single image
# -----------------------------
def run_on_single_image(image_path, output_folder):
    gc.collect()
    if not os.path.isfile(image_path):
        print(f"ERROR: File not found: {image_path}")
        return
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.exif_transpose(image)
    image = np.array(image)
    aug_img = apply_augmentation(image)
    debug_preview(image.astype(np.uint8), aug_img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    output_path = os.path.join(output_folder, f"aug_{os.path.basename(image_path)}")
    Image.fromarray(aug_img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save(output_path)
    print(f"Saved: {output_path}")
    gc.collect()

def debug_preview(original_img: np.ndarray, augmented_img: np.ndarray):
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    ax2 = fig.add_subplot(122)
    ax2.imshow(augmented_img)
    ax2.set_title("Augmented")
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

# -----------------------------
# Mode: Folder
# -----------------------------
def run_on_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for i, file in enumerate(files):
        print(f"[{i+1}/{len(files)}] Augmenting {file}")
        image_path = os.path.join(input_folder, file)
        run_on_single_image(image_path, output_folder)

# -----------------------------
# Main entry point
# -----------------------------
if __name__ == "__main__":
    image_path = "temp/data/Test_data_02/frame_00088_jpg.rf.4b23ed41569aea13427f34c50d0996ec_90CCW.jpg"
    folder_path = "temp/data/Test_data_02"
    output_folder = "temp/data/augmented"
    os.makedirs(output_folder, exist_ok=True)
    run_on_single_image(image_path, output_folder)
    # run_on_folder(folder_path, output_folder)

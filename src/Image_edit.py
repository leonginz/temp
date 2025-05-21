import cv2
import os

# === Configuration ===
folder_path = "datasets/swim_test_data"  # Replace with your path
image_extensions = ('.jpg', '.jpeg', '.png')
rotation_CW = cv2.ROTATE_90_CLOCKWISE                    # Rotation mode
rotation_CCW = cv2.ROTATE_90_COUNTERCLOCKWISE
# === Function ===
def rotate_images_in_folder(folder_path, extensions, rotation,suffix):
    """
    Rotates all images in the folder by a given angle and saves them with a suffix.

    Args:
        folder_path (str): Path to the image directory.
        extensions (tuple): Valid image file extensions.
        rotation: cv2 rotation constant (e.g., ROTATE_90_CLOCKWISE).
        suffix (str): Suffix to append to new filenames.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"⚠️ Could not load image: {filename}")
                continue

            rotated = cv2.rotate(image, rotation)

            name, ext = os.path.splitext(filename)
            new_filename = f"{name}{suffix}{ext}"
            new_path = os.path.join(folder_path, new_filename)

            cv2.imwrite(new_path, rotated)
            print(f"✅ Saved rotated image: {new_filename}")

# === Execute ===
rotate_images_in_folder(folder_path, image_extensions, rotation_CW,"_CW")
rotate_images_in_folder(folder_path, image_extensions, rotation_CCW,"_CCW")

import os

# === CONFIGURATION ===
IMAGE_DIR = "datasets/data_R/images"  # Update with your actual image folder
LABEL_DIR = "datasets/data_R/labels"  # Update with your actual label folder

def find_unmatched_files(image_dir, label_dir):
    # Get base names without extension
    image_basenames = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    label_basenames = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith('.txt')}

    images_without_labels = image_basenames - label_basenames
    labels_without_images = label_basenames - image_basenames

    print("🟥 Images without labels:")
    for name in sorted(images_without_labels):
        print(f"{name}.jpg")

    print("\n🟦 Labels without images:")
    for name in sorted(labels_without_images):
        print(f"{name}.txt")

# Run
find_unmatched_files(IMAGE_DIR, LABEL_DIR)

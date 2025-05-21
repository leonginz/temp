import os
import shutil
import random
from pathlib import Path
import time

def split_dataset_generic(root_dir, split=(70, 25, 5), seed=None):
    assert sum(split) == 100, "Split values must sum to 100"

    image_exts = ['.jpg', '.jpeg', '.png']
    image_files = []
    label_files = {}

    # Traverse all subdirectories to collect image and label files
    for root, _, files in os.walk(root_dir):
        for file in files:
            full_path = Path(root) / file
            if full_path.suffix.lower() in image_exts:
                image_files.append(full_path)
            elif full_path.suffix.lower() == '.txt':
                label_files[full_path.stem] = full_path

    # Create image-label pairs
    image_label_pairs = []
    for img_path in image_files:
        lbl_path = label_files.get(img_path.stem)
        if lbl_path:
            image_label_pairs.append((img_path, lbl_path))

    total = len(image_label_pairs)
    print(f"✅ Found {total} valid image-label pairs")
    if not total:
        print("⚠️ No matching image-label pairs found.")
        return

    # Seed random for reproducibility
    if seed is None:
        seed = int(time.time())
    print(f"🔀 Using random seed: {seed}")
    random.seed(seed)
    random.shuffle(image_label_pairs)

    train_end = int(total * split[0] / 100)
    val_end = train_end + int(total * split[1] / 100)

    splits = {
        'train': image_label_pairs[:train_end],
        'val': image_label_pairs[train_end:val_end],
        'test': image_label_pairs[val_end:]
    }

    # Define new output structure
    output_image_base = Path(root_dir) / 'images'
    output_label_base = Path(root_dir) / 'labels'

    for subset in splits:
        (output_image_base / subset).mkdir(parents=True, exist_ok=True)
        (output_label_base / subset).mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in splits[subset]:
            new_img_path = output_image_base / subset / img_path.name
            new_lbl_path = output_label_base / subset / lbl_path.name

            shutil.move(img_path, new_img_path)
            shutil.move(lbl_path, new_lbl_path)

    print("📊 Split summary:")
    for subset in splits:
        print(f" - {subset}: {len(splits[subset])} samples")


# === USAGE ===
split_dataset_generic(
    root_dir="datasets/Rotate_test/data_R",  # Folder that contains all images and labels
    split=(75, 21, 4)
)

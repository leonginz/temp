import os
import cv2
import math
import numpy as np

# === CONFIGURATION ===
IMAGE_DIR = "datasets/data_R/images"             # Input image folder
LABEL_DIR = "datasets/data_R/labels"             # Input label folder
OUTPUT_IMAGE_DIR = "datasets/data_R/rotated/images"
OUTPUT_LABEL_DIR = "datasets/data_R/rotated/labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def rotate_image_and_label(image_path, label_path, output_image_path, output_label_path, degrees=90, direction='CW'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    angle = -degrees if direction.upper() == 'CW' else degrees
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions of the image
    abs_cos = abs(rot_matrix[0, 0])
    abs_sin = abs(rot_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account translation
    rot_matrix[0, 2] += new_w / 2 - center[0]
    rot_matrix[1, 2] += new_h / 2 - center[1]

    rotated_image = cv2.warpAffine(image, rot_matrix, (new_w, new_h))
    cv2.imwrite(output_image_path, rotated_image)

    if not os.path.exists(label_path):
        print(f"Label file not found for {image_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        kpts = list(map(float, parts[5:]))

        # Convert bbox center to absolute coords
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h

        # Rotate bbox center
        new_coords = np.dot(rot_matrix, np.array([abs_cx, abs_cy, 1]))
        new_abs_cx, new_abs_cy = new_coords[0], new_coords[1]

        # Normalize new bbox values to new image dims
        new_cx = new_abs_cx / new_w
        new_cy = new_abs_cy / new_h
        new_bw = abs_bh / new_w if degrees % 180 != 0 else abs_bw / new_w
        new_bh = abs_bw / new_h if degrees % 180 != 0 else abs_bh / new_h

        # Rotate keypoints
        new_kpts = []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i], kpts[i + 1], int(kpts[i + 2])
            if v == 0:
                new_kpts.extend([x, y, v])
            else:
                abs_x = x * w
                abs_y = y * h
                rot_kpt = np.dot(rot_matrix, np.array([abs_x, abs_y, 1]))
                new_x = rot_kpt[0] / new_w
                new_y = rot_kpt[1] / new_h
                new_kpts.extend([new_x, new_y, v])

        kp_str = " ".join([f"{v:.6f}" if i % 3 != 2 else str(int(v)) for i, v in enumerate(new_kpts)])
        new_line = f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f} {kp_str}\n"
        new_lines.append(new_line)

    with open(output_label_path, 'w') as f:
        f.writelines(new_lines)

def run_rotation_augmentation(degrees, direction):
    SUFFIX = f"_{degrees}{direction.upper()}"

    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(IMAGE_DIR, filename)
            lbl_path = os.path.join(LABEL_DIR, name + '.txt')
            out_img_path = os.path.join(OUTPUT_IMAGE_DIR, name + SUFFIX + '.jpg')
            out_lbl_path = os.path.join(OUTPUT_LABEL_DIR, name + SUFFIX + '.txt')
            rotate_image_and_label(img_path, lbl_path, out_img_path, out_lbl_path, degrees=degrees, direction=direction)

    print("✅ Rotation augmentation complete.")

# Run
run_rotation_augmentation(degrees=90, direction='CW')
run_rotation_augmentation(degrees=90, direction='CCW')

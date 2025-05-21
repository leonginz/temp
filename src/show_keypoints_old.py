import cv2
import os

# CONFIG - change these as needed
IMAGE_DIR = 'datasets/data_R/rotated/images'             # folder containing .jpg images
LABEL_DIR = 'datasets/data_R/rotated/labels/'             # folder containing .txt labels
OUTPUT_DIR = 'datasets/data_R/rotatedpip /annotation_folder'      # where annotated images will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(IMAGE_DIR):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt'))
    output_path = os.path.join(OUTPUT_DIR, filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Could not load image: {image_path}")
        continue

    if not os.path.exists(label_path):
        print(f"[!] No label found for: {filename}")
        continue

    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        label_line = f.readline().strip().split()

    if len(label_line) < 5:
        print(f"[!] Label for {filename} has insufficient data")
        continue

    class_id = int(label_line[0])
    box_x, box_y, box_w, box_h = map(float, label_line[1:5])

    x0 = int((box_x - box_w/2) * w)
    x1 = int((box_x + box_w/2) * w)
    y0 = int((box_y - box_h/2) * h)
    y1 = int((box_y + box_h/2) * h)

    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(image, f'Class {class_id}', (x0, max(y0 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    kpt_data = list(map(float, label_line[5:]))
    if len(kpt_data) % 3 != 0:
        print(f"[!] Keypoints in {filename} not multiple of 3")
        continue

    num_keypoints = len(kpt_data) // 3
    for i in range(num_keypoints):
        kx, ky, kv = kpt_data[i*3:i*3+3]
        px = int(kx * w)
        py = int(ky * h)
        color = (0, 255, 0) if kv == 2 else (0, 165, 255)
        cv2.circle(image, (px, py), 5, color, -1)
        cv2.putText(image, str(i), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, image)
    print(f"[✓] Saved annotated image: {output_path}")

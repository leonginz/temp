import os

# === CONFIGURATION ===
LABEL_DIR = "datasets/coco_data_R/Swim/labels_02"  # Update this to your directory path

def clean_invisible_keypoints(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        class_and_box = parts[:5]
        keypoints = parts[5:]

        cleaned_kpts = []
        for i in range(0, len(keypoints), 3):
            x, y, v = float(keypoints[i]), float(keypoints[i+1]), float(keypoints[i+2])
            if v == 0.0:
                cleaned_kpts.extend(['0.000', '0.000', '0.000'])
            else:
                cleaned_kpts.extend([f"{x:.6f}", f"{y:.6f}", f"{v:.6f}"])

        new_line = ' '.join(class_and_box + cleaned_kpts) + '\n'
        new_lines.append(new_line)

    with open(label_file, 'w') as f:
        f.writelines(new_lines)

def run_cleaning():
    for file in os.listdir(LABEL_DIR):
        if file.endswith('.txt'):
            full_path = os.path.join(LABEL_DIR, file)
            clean_invisible_keypoints(full_path)
    print("✅ Keypoints cleaned.")

# Run it
run_cleaning()

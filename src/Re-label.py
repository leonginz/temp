import os
import glob

# If you have multiple classes, or multi-line .txt, adapt as needed.
# This script expects 1 line per label file:
#   class cx cy w h
#   + 14*(x y v)
# It will output 1 line with:
#   class cx cy w h
#   + 17*(x y v) (COCO style)

OLD_LABELS_DIR = 'datasets/coco_data_R/swim/labels/'
NEW_LABELS_DIR = 'datasets/coco_data_R/swim/new_labels/'
os.makedirs(NEW_LABELS_DIR, exist_ok=True)

# Mapping from your 14 keypoints → COCO index
REMAP = {
    0: 16,  # right ankle
    1: 14,  # right knee
    2: 12,  # right hip
    3: 15,  # left ankle
    4: 13,  # left knee
    5: 1,   # left eye
    6: 6,   # right shoulder
    7: 8,   # right elbow
    8: 10,  # right wrist
    9: 5,   # left shoulder
    10: 7,  # left elbow
    11: 9,  # left wrist
    12: 11, # left hip
    13: 2   # right eye
}
NUM_COCO_KPTS = 17

for txt_file in glob.glob(os.path.join(OLD_LABELS_DIR, '*.txt')):
    with open(txt_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            continue  # empty file

    parts = line.split()
    if len(parts) < 5 + 14*3:
        # 5 = class + box
        # 14*3 = old kpts
        print(f"Skipping {txt_file}, not enough floats.")
        continue

    # parse class & box
    cls_id = parts[0]
    box_x = parts[1]
    box_y = parts[2]
    box_w = parts[3]
    box_h = parts[4]

    # parse old keypoints (14 sets)
    old_kpts = parts[5:]
    old_kpts = list(map(float, old_kpts))  # convert to float

    # We'll build a 51-length array for 17 kpts (x,y,v)
    new_kpts = [0.0]*(NUM_COCO_KPTS*3)

    # place each old kpt into new positions
    for old_idx in range(14):
        x = old_kpts[old_idx*3 + 0]
        y = old_kpts[old_idx*3 + 1]
        v = old_kpts[old_idx*3 + 2]
        # map old_idx to coco_idx
        coco_idx = REMAP[old_idx]  # e.g., 0 → 16
        new_kpts[coco_idx*3 + 0] = x
        new_kpts[coco_idx*3 + 1] = y
        new_kpts[coco_idx*3 + 2] = v

    # Now we have a 51 float array in the correct order for COCO
    # If you want to fill missing ones differently, you can do so here

    # Build output line: class, box, 17*(x y v)
    out_parts = [cls_id, box_x, box_y, box_w, box_h]
    out_parts += [f"{val:.6f}" for val in new_kpts]

    out_line = " ".join(out_parts)

    # Save to new .txt
    base_name = os.path.basename(txt_file)
    new_path = os.path.join(NEW_LABELS_DIR, base_name)
    with open(new_path, 'w') as f_out:
        f_out.write(out_line + "\n")

    print(f"Wrote {new_path}")

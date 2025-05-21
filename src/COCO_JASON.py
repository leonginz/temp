import os
import json

def yolo_pose_to_coco(
    yolo_txt_path,
    output_json_path="output.json",
    image_filename="my_image.jpg",
    image_width=1920,
    image_height=1080
):
    """
    Converts a single YOLO pose annotation (with 17 keypoints) into
    a minimal COCO pose annotation file.

    Args:
        yolo_txt_path (str): Path to your YOLO pose .txt file (one line).
        output_json_path (str): Where to save the resulting COCO .json file.
        image_filename (str): The name of your image file (COCO "file_name").
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
    """
    # 1. Read YOLO .txt line
    with open(yolo_txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) == 0:
        raise ValueError("No annotation lines found in the YOLO .txt file.")
    # For this script, assume only one bounding box/person per file:
    annotation_line = lines[0].split()

    # 2. Parse bounding box in YOLO format
    # class cx cy w h
    cls = int(float(annotation_line[0]))  # might be "0.0" if it was float
    cx = float(annotation_line[1])
    cy = float(annotation_line[2])
    bw = float(annotation_line[3])
    bh = float(annotation_line[4])

    # Convert YOLO (cx, cy, w, h) -> COCO (x_min, y_min, width, height)
    x_min = (cx - bw/2) * image_width
    y_min = (cy - bh/2) * image_height
    abs_w = bw * image_width
    abs_h = bh * image_height
    area = abs_w * abs_h  # approximate area

    # 3. Parse keypoints (17 sets)
    # Each set => x, y, v
    # all normalized, so we convert to absolute pixel coords
    keypoints_floats = list(map(float, annotation_line[5:]))  # after box
    if len(keypoints_floats) % 3 != 0:
        raise ValueError("Keypoint data doesn't appear to be in x,y,v triplets.")

    # Flatten into [x1, y1, v1, x2, y2, v2, ...]
    coco_keypoints = []
    for i in range(0, len(keypoints_floats), 3):
        kp_x_norm = keypoints_floats[i]
        kp_y_norm = keypoints_floats[i+1]
        kp_v = keypoints_floats[i+2]

        # Convert normalized coords to absolute
        kp_x_abs = kp_x_norm * image_width
        kp_y_abs = kp_y_norm * image_height

        # Append to list
        coco_keypoints.extend([kp_x_abs, kp_y_abs, kp_v])

    num_keypoints = len(coco_keypoints) // 3

    # 4. Build minimal COCO structure
    coco_dict = {
        "info": {
            "description": "Example COCO Pose for a single image",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  # assuming 'person' category has id=1
                "bbox": [x_min, y_min, abs_w, abs_h],
                "area": area,
                "iscrowd": 0,
                "keypoints": coco_keypoints,
                "num_keypoints": num_keypoints
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                # Standard COCO 17 keypoints
                "keypoints": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ],
                # Pairs of keypoint indices that define the skeleton
                # (Following the official COCO order, 1-based)
                "skeleton": [
                    [1, 2], [1, 3], [2, 4], [3, 5],
                    [1, 6], [1, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [6, 12], [7, 13],
                    [12, 14], [13, 15], [14, 16], [15, 17]
                ]
            }
        ]
    }

    # 5. Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as out_file:
        json.dump(coco_dict, out_file, indent=4)

    print(f"Saved COCO pose JSON to {output_json_path}")

    return coco_dict


# ------------------------
# Example usage:
if __name__ == "__main__":
    # Path to your YOLO pose .txt (like your 07.txt)
    YOLO_TXT_PATH = "07.txt"
    # Output JSON path
    OUTPUT_JSON_PATH = "07_coco.json"

    # Hypothetical image info
    IMAGE_FILENAME = "07.jpg"   # or "frame_00007.jpg"
    IMAGE_WIDTH = 1280          # or your actual width
    IMAGE_HEIGHT = 720          # or your actual height

    yolo_pose_to_coco(
        yolo_txt_path=YOLO_TXT_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        image_filename=IMAGE_FILENAME,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )

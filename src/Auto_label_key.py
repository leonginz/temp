import os
import cv2
import torch
from ultralytics import YOLO


def confidence_to_visibility(conf):
    """
    Maps a keypoint confidence to a visibility integer.
    Feel free to adjust thresholds as desired.
    Example thresholds:
      - conf < 0.20 => Not detected => (x=0, y=0, v=0)
      - 0.20 <= conf < 0.50 => Partial => v=1
      - conf >= 0.50 => Confident => v=2
    """
    if conf < 0.20:
        return 0
    elif conf < 0.50:
        return 1
    else:
        return 2


def annotate_image_with_pose(
        image_path,
        model,
        save_txt=True,
        output_dir="outputs",
        conf_threshold=0.25
):
    """
    Loads an image, runs YOLO pose inference, and saves a .txt annotation
    in a format similar to your sample (class, bbox center/size, then 17 sets of x,y,v).
    Each numeric value is printed with 6-decimal precision.

    Args:
        image_path (str): Path to the image file.
        model (YOLO): Ultralytics YOLO model (pose version).
        save_txt (bool): If True, writes annotations to a .txt file.
        output_dir (str): Directory to save .txt outputs.
        conf_threshold (float): Confidence threshold for bounding-box predictions.

    Returns:
        list of str: Each element is a line of annotation for one bounding box in YOLO-like format.
    """
    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")
    h, w, _ = img.shape

    # 2. Inference (use 'model.predict' in YOLOv8)
    results = model.predict(img, conf=conf_threshold, verbose=False)
    detections = results[0]  # For a single image, results is a list of length 1

    # 3. Prepare output directory
    if save_txt:
        os.makedirs(output_dir, exist_ok=True)

    lines_to_write = []

    # 4. Parse each detection
    for box, kps in zip(detections.boxes, detections.keypoints):
        # --- Bounding box (xyxy) ---
        x1, y1, x2, y2 = box.xyxy[0]
        class_idx = int(box.cls[0]) if box.cls is not None else 0

        # Convert to YOLO (cx, cy, w, h) in normalized [0..1]
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # --- Keypoints ---
        # 'kps' is a Keypoints object. We'll extract:
        #  -> kps.xy  or kps.xyn (absolute vs. normalized)
        #  -> kps.conf for confidence
        # Then apply thresholds to produce (x, y, v).

        # If you want normalized coords for keypoints (like your sample),
        # use kps.xyn. If your sample used absolute coords, use kps.xy.
        # Your example looks normalized, so let's use xyn:
        kps_xy = kps.xyn[0].cpu().numpy()  # shape: (N, 2)
        kps_conf = kps.conf[0].cpu().numpy()  # shape: (N,)

        # Prepare final triplets [kp_x, kp_y, v]
        kp_values = []
        for (kp_x, kp_y), kp_score in zip(kps_xy, kps_conf):
            v = confidence_to_visibility(kp_score)
            # If v=0 => set x=0, y=0
            if v == 0:
                kp_x, kp_y = 0.0, 0.0

            kp_values.extend([kp_x, kp_y, float(v)])  # store v as float for consistent formatting

        # Combine into a single line:
        # class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v ...
        line_data = [class_idx, cx, cy, bw, bh] + kp_values

        # Create a space-separated string with 6-decimal formatting
        line_str = f"{int(line_data[0])} " + " ".join(f"{x:.6f}" for x in line_data[1:])
        lines_to_write.append(line_str)

    # 5. Save to .txt if needed
    if save_txt and len(lines_to_write) > 0:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            for line in lines_to_write:
                f.write(line + "\n")

    return lines_to_write


def main():
    # Replace with your pose model, e.g., "yolov8s-pose.pt"
    model_path = "yolov8s-pose.pt"
    model = YOLO(model_path)

    # Example usage:
    image_path = r"outputs/07.jpg"

    # Run annotation function
    annotation_lines = annotate_image_with_pose(
        image_path=image_path,
        model=model,
        save_txt=True,
        output_dir="outputs",  # or your desired directory
        conf_threshold=0.25  # adjust if you want more or fewer detections
    )

    print("\nGenerated Annotation Lines:")
    for line in annotation_lines:
        print(line)


if __name__ == "__main__":
    main()

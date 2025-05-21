#################################################
# CONFIGURATION
#################################################
import os

IMAGE_PATH = 'C:/Users/97254/PycharmProjects/pythonProject/Data set/Swim_detect_Data/train/images/frame_06300_jpg.rf.bd2e85cab78f98aa990a18b5542c8781.jpg'            # e.g., 'images/frame_08316.jpg'
LABELS_DIR = 'C:/Users/97254/PycharmProjects/pythonProject/Data set/Swim_detect_Data/train/labels/'              # e.g., 'labels/'
MODEL_PATH = 'yolov8n-pose.pt'

# Derive label path from image name
image_filename = os.path.basename(IMAGE_PATH)     # 'frame_08316.jpg'
label_filename = os.path.splitext(image_filename)[0] + '.txt'  # 'frame_08316.txt'
LABEL_PATH = os.path.join(LABELS_DIR, label_filename)

# Output file names
OUTPUT_PATH_FILE = 'annotated_from_file.jpg'
OUTPUT_PATH_MODEL = 'annotated_from_model.jpg'

#################################################
# IMPORTS
#################################################
import cv2
from ultralytics import YOLO


#################################################
# FUNCTION 1: Annotate from file
#################################################
def annotate_from_file(image_path: str, label_path: str, output_path: str = "annotated.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        label_line = f.readline().strip().split()

    if len(label_line) < 5:
        raise ValueError("Label file does not have enough floats (class + box + keypoints).")

    class_id = int(label_line[0])
    box_x = float(label_line[1])
    box_y = float(label_line[2])
    box_w = float(label_line[3])
    box_h = float(label_line[4])

    x0 = int((box_x - box_w / 2) * w)
    x1 = int((box_x + box_w / 2) * w)
    y0 = int((box_y - box_h / 2) * h)
    y1 = int((box_y + box_h / 2) * h)

    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(image, f"Class {class_id}", (x0, max(y0 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    kpt_data = list(map(float, label_line[5:]))
    if len(kpt_data) % 3 != 0:
        raise ValueError("Keypoint data is not a multiple of 3 (x, y, v).")

    num_keypoints = len(kpt_data) // 3
    for i in range(num_keypoints):
        kx = kpt_data[i * 3 + 0]
        ky = kpt_data[i * 3 + 1]
        kv = int(kpt_data[i * 3 + 2])

        px = int(kx * w)
        py = int(ky * h)
        color = (0, 255, 0) if kv == 2 else (0, 165, 255)

        cv2.circle(image, (px, py), 5, color, -1)
        cv2.putText(image, str(i), (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to {output_path}")
    cv2.imshow("Annotated from File", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#################################################
# FUNCTION 2: Annotate from model
#################################################
def annotate_from_model(image_path: str, model_path: str, output_path: str = "annotated_model.jpg"):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.25)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    for result in results:
        boxes = result.boxes
        kpts_tensor = result.keypoints  # Keypoints object

        for i, box in enumerate(boxes):
            c_id = int(box.cls[0]) if box.cls is not None else -1
            xywh = box.xywh[0].cpu().numpy()
            (cx, cy, bw, bh) = xywh

            x0 = int(cx - bw / 2)
            y0 = int(cy - bh / 2)
            x1 = int(cx + bw / 2)
            y1 = int(cy + bh / 2)

            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(image, f"Class {c_id}", (x0, max(y0 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if kpts_tensor is not None and kpts_tensor.xy is not None:
                keypoints = kpts_tensor.xy[i].cpu().numpy()  # shape: (K, 2)
                for kp_idx, (kx, ky) in enumerate(keypoints):
                    px, py = int(kx), int(ky)
                    color = (0, 255, 0)
                    cv2.circle(image, (px, py), 5, color, -1)
                    cv2.putText(image, str(kp_idx), (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image (from model) to {output_path}")
    cv2.imshow("Annotated from Model", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#################################################
# MAIN
#################################################
def main():
    # annotate_from_file(IMAGE_PATH, LABEL_PATH, OUTPUT_PATH_FILE)
    # annotate_from_model(IMAGE_PATH, MODEL_PATH, OUTPUT_PATH_MODEL)
    annotate_from_model('C:/Users/97254/Pictures/YOLO Project/Test/person.jpg', MODEL_PATH, OUTPUT_PATH_MODEL)


if __name__ == "__main__":
    main()

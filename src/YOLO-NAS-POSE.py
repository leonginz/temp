import torch
import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO-NAS pose model
model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").to(device)

# Open video file
video_path = "pool_test.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose estimation
    results = model.predict(frame, iou=0.5, conf=0.3)

    # Draw results
    if results.prediction is not None:
        annotated_frame = results.draw()
        cv2.imshow("YOLO-NAS Pose", annotated_frame)

        # Optional: Extract keypoints
        if len(results.prediction.keypoints) > 0:
            keypoints = results.prediction.keypoints[0].xy.cpu().numpy()
            print("Keypoints:", keypoints)
    else:
        cv2.imshow("YOLO-NAS Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import torch
import cv2
from ultralytics import YOLO
from ultralytics import NAS

torch.cuda.empty_cache()
# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLO pose model on GPU
model = YOLO("yolo11l-pose.pt").to(device)

# # Load Trained Model
# model = YOLO('runs/pose/train3/weights/best.pt')

# # Load the YOLO-nas pose model on GPU
# model = NAS("yolo_nas_s.pt").to(device)

video_path = "pool_test.mp4"  # Your video file path
cap = cv2.VideoCapture(video_path)

frame_count = 0
skip_frames = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    frame_count += 1
    # Skip frames: process only when frame_count is divisible by skip_frames
    if frame_count % skip_frames != 0:
        continue

    # Resize frame to 640x640 (model input size)
    frame_resized = cv2.resize(frame, (640, 640))
    # Convert frame to tensor in BCHW format and normalize to [0,1]
    frame_tensor = (
        torch.from_numpy(frame_resized)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device) / 255.0
    )

    # Run YOLO inference (using tracking method)
    results = model.predict(frame_tensor, show=True, save=False, persist=True, verbose=False)
    print(model.device)

    # Optionally, extract keypoints (if available)
    if results and hasattr(results[0], 'keypoints'):
        results2 = results[0].keypoints.xy.cpu().numpy()

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

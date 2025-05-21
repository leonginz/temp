import os
import cv2
import torch
import logging
from ultralytics import YOLO

# --- Suppress YOLO's default console prints ---
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLO model on GPU
model = YOLO("runs/pose/train3/weights/best.pt").to(device)

# Input video path
video_path = "downloads/vid02.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get total number of frames for progress calculation
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create directory to save cropped images
output_dir = "cropped_images"
os.makedirs(output_dir, exist_ok=True)

frame_index = 0  # actual index in video
saved_frame_count = 0
skip_interval = 5  # Process every 5th frame (change as needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    if frame_index % skip_interval == 0:
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

        # Run YOLO inference
        results = model(frame_tensor)

        # Check if any bounding box has confidence > xxx
        save_frame = False
        for result in results:
            if result.boxes is not None and result.boxes.conf is not None:
                for conf in result.boxes.conf:
                    if  conf > 0.6:  # Confidence threshold
                        save_frame = True
                        break

        # Save frame if confidence condition is met
        if save_frame:
            output_path = os.path.join(output_dir, f"frame_{frame_index:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

        # Show progress only for processed frames
        progress = (frame_index / total_frames) * 100
        print(f"Processing progress: {progress:.2f}%")

    frame_index += 1  # always increment

# Release video capture and print summary
cap.release()
print(f"Processing complete. {saved_frame_count} frames saved to '{output_dir}'.")

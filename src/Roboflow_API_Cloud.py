import cv2
import numpy as np
import tempfile
import os
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow HTTP client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="kuhJrQ5WfNq25aF5qYmb"
)

# Confidence threshold
confidence_threshold = 0.2

# Open video
cap = cv2.VideoCapture("downloads/vid01.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_filename = tmp.name
        cv2.imwrite(temp_filename, frame)

    # Send frame to Roboflow model
    result = CLIENT.infer(temp_filename, model_id="swimming_detection/1")

    # Clean up temp file
    os.remove(temp_filename)

    # Parse predictions
    predictions = result.get("predictions", [])
    for p in predictions:
        if p["confidence"] >= confidence_threshold:
            print(f"Identified: {p['class']} with confidence {p['confidence']:.2f}")
            x0, y0 = int(p["x"] - p["width"] / 2), int(p["y"] - p["height"] / 2)
            x1, y1 = int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)
            label = f"{p['class']} {p['confidence']:.2f}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    # Show frame
    cv2.imshow("Predictions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
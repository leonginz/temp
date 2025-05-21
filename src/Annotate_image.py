import os
import cv2
from ultralytics import YOLO

# Load a pre-trained YOLO Pose model
# model = YOLO("runs/pose_retrain_phase02/weights/best.pt")
# model = YOLO("yolo11m-pose.pt")
model = YOLO("yolo11x-pose.pt")
# model  = YOLO("yolo-runs-no-freeze/pose_train2/weights/best.pt")

# Define input and output directories
input_folder = "datasets/data_R/images"  # Replace with your folder containing images
output_folder = "annotated_images"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(input_folder, image_file)

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Run YOLO Pose inference on the image
    results = model.predict(image)

    # Annotate the image with bounding boxes, labels, and keypoints
    annotated_image = results[0].plot()

    # Save the annotated image to the output folder
    output_path = os.path.join(output_folder, f"annotated_{image_file}")
    cv2.imwrite(output_path, annotated_image)
    print(f"Saved annotated image: {output_path}")

    # Optional: Extract and print keypoints for each detected person
    if results[0].keypoints is not None:
        for i, keypoints in enumerate(results[0].keypoints.xy):
            print(f"Keypoints for person {i + 1} in {image_file}:")
            print(keypoints.cpu().numpy())  # Convert keypoints to numpy array for easier handling

print("Annotation complete. All images have been processed and saved.")
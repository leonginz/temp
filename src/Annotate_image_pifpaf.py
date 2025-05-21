import os
import openpifpaf
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive rendering
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import sys 

# Define input and output directories
input_folder = "temp/data/Test_data_02"  # Replace with your folder containing images
output_folder = "temp/data/annotated_images"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def load_image(image_path):
    """Load and convert an image to a numpy array."""
    return np.asarray(PIL.Image.open(image_path).convert('RGB'))


def run_pifpaf_prediction(image, checkpoint='resnet50'):
    """Run OpenPifPaf prediction with custom decoding options (legacy-compatible)."""

    # Configure decoder settings
    openpifpaf.decoder.CifCaf.instance_threshold = 0.07
    openpifpaf.decoder.CifCaf.seed_threshold = 0.1
    openpifpaf.decoder.CifCaf.keypoint_threshold = 0.05
    openpifpaf.decoder.CifCaf.force_complete_pose = True
    openpifpaf.decoder.CifCaf.orientation_invariant = True
    openpifpaf.decoder.CifCaf.dense_connections = True
    openpifpaf.decoder.CifCaf.apply_soft_nms = True

    predictor = openpifpaf.Predictor(checkpoint=checkpoint)
    predictions = predictor.numpy_image(np.asarray(image))  
    return predictions


def annotate_image_with_pifpaf(image, predictions, confidence_threshold=0.02):
    """Manually annotate keypoints with confidence filtering."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for ann in predictions[0]:  # predictions[0] is a list of Annotation objects
        keypoints = ann.data  # shape (17, 3) for COCO format: (x, y, confidence)
        for i, (x, y, v) in enumerate(keypoints):
            if v >= confidence_threshold:
                ax.scatter(x, y, s=40, c='red')
                ax.text(x + 2, y, f'{i}:{v:.2f}', fontsize=6, color='yellow')  # Label with ID and confidence

    ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    annotated_image = buf.reshape((h, w, 4))[..., :3]  # Convert RGBA → RGB
    plt.close(fig)
    return annotated_image



# 
# def annotate_image_with_pifpaf(image, predictions, confidence_threshold=0.1):
#     """Manually annotate keypoints with confidence filtering."""
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image)

#     for ann in predictions[0]:  # predictions[0] is a list of Annotation objects
#         keypoints = ann.data  # shape (17, 3) for COCO format: (x, y, confidence)
#         for i, (x, y, v) in enumerate(keypoints):
#             if v >= confidence_threshold:
#                 ax.scatter(x, y, s=40, c='red')
#                 ax.text(x + 2, y, f'{i}:{v:.2f}', fontsize=6, color='yellow')  # Label with ID and confidence

#     ax.axis('off')
#     plt.tight_layout()

#     fig.canvas.draw()
#     buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#     w, h = fig.canvas.get_width_height()
#     annotated_image = buf.reshape((h, w, 4))[..., :3]  # Convert RGBA → RGB
#     plt.close(fig)
#     return annotated_image
    



# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(input_folder, image_file)

    # Load the image
    image = load_image(image_path)

    # Run PifPaf prediction
    predictions = run_pifpaf_prediction(image)

    # Annotate the image
    annotated_image = annotate_image_with_pifpaf(image, predictions)

    # Save the annotated image to the output folder
    output_path = os.path.join(output_folder, f"annotated_{image_file}")
    PIL.Image.fromarray(annotated_image).save(output_path)
    print(f"Saved annotated image: {output_path}")

print("Annotation complete. All images have been processed and saved.")
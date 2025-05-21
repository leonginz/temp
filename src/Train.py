from ultralytics import YOLO

def train_model():
    # Load the pre-trained YOLOv8 pose model
    model = YOLO('yolov8n-pose.pt')  # You can replace with yolov8s/m/l/x-pose.pt

    # Start training
    results = model.train(
        data='datasets/coco_data_sample/data.yaml',  # Path to your YAML
        epochs=20,                                    # Change as needed
        imgsz=640,
        batch=2,
        freeze=10,                                   # Freeze first 10 layers
        name="pose_train",
        # lr0=0.0001,
        # lrf=0.0001,
        project="yolo-runs-no-freeze"
    )

    # Optional: print results or save summary
    print("Training completed!")
    print(results)

# Required for Windows multiprocessing
if __name__ == '__main__':
    train_model()



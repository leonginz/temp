from ultralytics import YOLO

def train():
    model = YOLO("yolov8n-pose.pt")
    model.train(
        data='datasets/coco_data_sample/data.yaml',
        epochs=1,
        imgsz=640,
        batch=2,
        name="train5"
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # <-- safe for Windows
    train()

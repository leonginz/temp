from ultralytics import YOLO, checks, hub

def train_model():
    checks()
    hub.login('6defb11c6e7d2fc09b7e9a3c0897efa001e963260b')

    model = YOLO('https://hub.ultralytics.com/models/Ek7ArbiWK3KzgXJ6Ntly')

    results = model.train(
        data='datasets/swim_d01.yaml',  # required if not already in the model
        epochs=30,                 # or however many you want
        imgsz=640
    )

if __name__ == "__main__":
    train_model()

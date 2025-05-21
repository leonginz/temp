from ultralytics import YOLO
import cv2


model = YOLO("yolo11s-pose.pt")

# # Load Trained Model
# model = YOLO('runs/pose/train3/weights/best.pt')

video_path = "downloads/vid01.mp4"  # Change this to your actual file path
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # results = model.track(frame, show=True, save=False, persist=False, verbose=False)
    results = model.predict(frame, show=True, save=False, verbose=False)


    print(model.device)
    # results2 = results[0].keypoints.xy.cpu().numpy()
    nose = ''
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()

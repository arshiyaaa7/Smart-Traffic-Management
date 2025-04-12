import cv2
from google.colab.patches import cv2_imshow
from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')
image_folder = '/content/data/train/Accident'

def is_intersecting(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))][:10]

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)

    results = model(frame)
    vehicles = []

    for r in results:
        for det in r.boxes:
            x1, y1, x2, y2 = int(det.xyxy[0][0]), int(det.xyxy[0][1]), int(det.xyxy[0][2]), int(det.xyxy[0][3])
            cls = int(det.cls[0])
            if cls in [2, 3, 5, 7]:  # Cars, motorbikes, buses, trucks
                vehicles.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    crash_detected = False
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if is_intersecting(vehicles[i], vehicles[j]):
                x1, y1, x2, y2 = vehicles[i]
                x3, y3, x4, y4 = vehicles[j]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                cv2.putText(frame, 'Crash Detected!', (min(x1, x3), min(y1, y3) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                crash_detected = True

    print(f"📷 Processed Image: {image_name}")
    if crash_detected:
        print(f"🚨 Crash detected in image: {image_name}")
    cv2_imshow(frame)
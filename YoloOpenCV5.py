import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Webcam
cap = cv2.VideoCapture(0)  #   0 is the default camera index

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Frame capture loop
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break  # Break the loop if frame is not captured

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image
    frame_pil = Image.fromarray(frame_rgb)

    # Inference
    results = model(frame_pil, size=640)  # Single image

    # Draw bounding boxes on the frame
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,   255,   0),   2)
        label = f'{results.names[int(cls)]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 -   10), cv2.FONT_HERSHEY_SIMPLEX,   0.5, (0,   255,   0),   2)

    cv2.imshow('YOLOv5 Live Detection', frame)

    if cv2.waitKey(1) &   0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

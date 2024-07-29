from ultralytics import YOLO

model = YOLO('yolov5s.pt')
#model = YOLO('yolov5s.pt')

# Perform pose estimation 0 is default webcam input
results = model(source=0, show=True, conf=0.3, save=False)

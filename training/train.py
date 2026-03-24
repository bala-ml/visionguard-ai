from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='training/data.yaml',
    epochs=80,
    imgsz=1280,
    batch=4,
    name='tyre_detector_desert'
)
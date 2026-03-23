from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='training/data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='tyre_detector'
)
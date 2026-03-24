from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='training/data.yaml',
    epochs=70,          
    imgsz=640,
    batch=8,
    patience=15,        
    name='tyre_detector_desert'
)
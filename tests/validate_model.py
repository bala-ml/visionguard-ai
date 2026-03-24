from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("../models/best.pt")

test_folder = Path("../dataset/test/images")

output_folder = Path("../tests/validation_output")
output_folder.mkdir(exist_ok=True)

print("Starting Validation...\n")

for img_path in test_folder.glob("*.*"):

    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    results = model(frame, conf=0.25)

    annotated = results[0].plot()

    detections = len(results[0].boxes) if results[0].boxes is not None else 0

    print(f"{img_path.name} → Tyres detected: {detections}")

    save_path = output_folder / img_path.name
    cv2.imwrite(str(save_path), annotated)

print("\n Validation Completed!")
print(f"Results saved in: {output_folder}")
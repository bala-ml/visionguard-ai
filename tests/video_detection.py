from ultralytics import YOLO
import cv2
from pathlib import Path

# ⭐ Load trained model
model = YOLO("../models/best.pt")

# ⭐ Video path
video_path = Path(__file__).resolve().parents[1] / "assets" / "sample_videos" / "test.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ⭐ ROI (wider area = better detection)
    roi = frame[int(h*0.2):h, int(w*0.3):w]

    # ⭐ DETECTION (not tracking)
    results = model.predict(
        roi,
        conf=0.25,        # Lower = detect more objects
        imgsz=1280,       # Higher = better small-object detection
        device="cpu"
    )

    annotated_roi = results[0].plot()

    # ⭐ Put ROI back into frame
    frame[int(h*0.2):h, int(w*0.3):w] = annotated_roi

    cv2.imshow("Tyre Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
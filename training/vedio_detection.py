from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("../models/best.pt")

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

    # ⭐ ROI — adjust based on your video
    roi = frame[int(h*0.3):h, int(w*0.4):w]

    # ⭐ TRACKING ON ROI
    results = model.track(
        roi,
        persist=True,
        conf=0.25,
        imgsz=960
    )

    annotated_roi = results[0].plot()

    # ⭐ Put ROI back into original frame
    frame[int(h*0.3):h, int(w*0.4):w] = annotated_roi

    # ⭐ FIX BLUE ISSUE — convert for display only
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("Tyre Tracking", display_frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import os
import sys
import streamlit as st
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["DISPLAY"] = ":0"
import cv2 as cv2_headless
sys.modules['cv2'] = cv2_headless
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
* {
    font-family: 'Poppins', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "best.pt"
    return YOLO(str(model_path))

model = load_model()

base_path = Path(__file__).resolve().parents[1] / "assets" / "sample_videos"

video_options = {
    "Desert Scenario": base_path / "test.mp4",
    "Mountain Scenario": base_path / "test2.mp4"
}

st.title("🛡️ VisionGuard AI")
st.subheader("Universal Video Intelligence System")

st.markdown(
    """
    Detect and analyze objects in videos using AI.  
    This demo focuses on **tyre detection in real-world conditions**.
    """
)

st.divider()

st.sidebar.header("Settings")

confidence = 0.45

selected_video_name = st.sidebar.selectbox(
    "Select Scenario",
    list(video_options.keys())
)

video_path = video_options[selected_video_name]

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Original Video")

    st.video(str(video_path))

with col2:
    st.header("Detection Output")

    run_button = st.button("▶ Run Detection")

    if run_button:

        cap = cv2.VideoCapture(str(video_path))
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)

            annotated_frame = results[0].plot()

            stframe.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

st.divider()

# Info
st.header("📊 System Information")

c1, c2, c3 = st.columns(3)

c1.metric("Model", "YOLOv8")
c2.metric("Task", "Object Detection")
c3.metric("Target", "Tyre")

st.markdown(
    """
    ---
    Built for AI-based video analytics demonstration  
    VisionGuard AI — Real-time Detection System
    """
)
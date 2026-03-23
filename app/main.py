import streamlit as st

# Page settings
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

# ===== HEADER =====
st.title("🛡️ VisionGuard AI")
st.subheader("Universal Video Intelligence System")

st.markdown(
    """
    Detect and analyze objects in videos using AI.
    This demo focuses on **tyre detection in real-world conditions**.
    """
)

st.divider()

# ===== SIDEBAR =====
st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.info(
    "Adjust detection sensitivity."
)

# ===== MAIN LAYOUT =====
col1, col2 = st.columns([1, 1])

# LEFT — Upload Section
with col1:
    st.header("📤 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        st.success("Video uploaded successfully!")

        st.write("File name:", uploaded_file.name)
        st.write("File size:", round(uploaded_file.size / (1024*1024), 2), "MB")

        st.video(uploaded_file)

# RIGHT — Output Section
with col2:
    st.header("Detection Output")

    st.info(
        "Detection results will appear here after processing."
    )

    st.empty()  # Placeholder for future output

st.divider()

# ===== PROJECT INFO =====
st.header("📊 System Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("Model", "YOLOv8")

with info_col2:
    st.metric("Task", "Object Detection")

with info_col3:
    st.metric("Target", "Tyre")

# ===== FOOTER =====
st.markdown(
    """
    ---
    Built for AI-based video analytics demonstration  
    VisionGuard AI — Real-time Detection System
    """
)
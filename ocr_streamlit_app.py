import streamlit as st
import torch
import easyocr
import cv2
import time
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import tempfile
import os

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def process_video(video_path, reader, frame_skip=5, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)

    start_time = time.time()
    frame_count = 0
    ocr_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (width, height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = reader.readtext(rgb_frame)
            ocr_results.append(result)

        frame_count += 1

    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time

    cap.release()

    return avg_fps, ocr_results

def display_fps_comparison(gpu_fps, cpu_fps):
    labels = ['GPU', 'CPU']
    fps_values = [gpu_fps, cpu_fps]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, fps_values, color=['blue', 'green'])
    ax.set_ylabel('Frames Per Second (FPS)')
    ax.set_title('GPU vs CPU FPS Comparison')
    ax.set_ylim(0, max(fps_values) + 2)
    st.pyplot(fig)

st.title("OCR Performance: GPU vs CPU")

if "video_key" not in st.session_state:
    st.session_state["video_key"] = 0  # Initialize with 0 and increment if needed

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], key=f"video_upload_{st.session_state['video_key']}")
frame_skip = st.slider("Frame Skip", min_value=1, max_value=10, value=5, key="frame_skip_slider")
resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="resize_factor_slider")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.text("Loading GPU model...")
    reader_gpu = easyocr.Reader(['en'], gpu=True)

    st.text("Processing video with GPU...")
    gpu_fps, gpu_results = process_video(video_path, reader_gpu, frame_skip=frame_skip, resize_factor=resize_factor)

    st.text("Loading CPU model...")
    reader_cpu = easyocr.Reader(['en'], gpu=False)

    st.text("Processing video with CPU...")
    cpu_fps, cpu_results = process_video(video_path, reader_cpu, frame_skip=frame_skip, resize_factor=resize_factor)

    if gpu_fps is not None and cpu_fps is not None:
        st.write(f"**GPU FPS:** {gpu_fps:.2f}")
        st.write(f"**CPU FPS:** {cpu_fps:.2f}")
        display_fps_comparison(gpu_fps, cpu_fps)
    else:
        st.error("Could not calculate FPS for one or both runs.")

    if gpu_results is not None and cpu_results is not None:
        st.write(f"\n**Accuracy comparison (sample results):**")
        for i in range(min(5, len(gpu_results))):
            gpu_text = ' '.join([item[1] for item in gpu_results[i]])
            cpu_text = ' '.join([item[1] for item in cpu_results[i]])
            st.write(f"\n**Frame {i + 1}:**")
            st.write(f"GPU OCR: {gpu_text}")
            st.write(f"CPU OCR: {cpu_text}")
            st.write(f"Similarity: {similarity(gpu_text, cpu_text):.2f}")
    else:
        st.error("Could not perform OCR accuracy comparison due to missing results.")

    os.remove(video_path)

    st.session_state["video_key"] += 1

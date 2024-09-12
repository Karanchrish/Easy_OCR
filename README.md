# Video OCR Performance Comparison
This project compares the performance of Optical Character Recognition (OCR) using EasyOCR on GPU and CPU. It processes a video file, measures the frames per second (FPS) for each method, and compares the OCR accuracy between the two.

## Features
 - Processes a video file with EasyOCR on both GPU and CPU.
 - Compares the FPS of GPU and CPU processing.
 - Compares OCR results from GPU and CPU.
 - Visualizes the FPS comparison using Matplotlib.
## Requirements
 - Python 3.x
 - PyTorch
 - EasyOCR
 - OpenCV
 - Matplotlib
## Usage
  - Prepare the Video File

    Place your video file in the video_path specified in the script or modify the video_path variable to point to your video file.

 - View Results

   - The script will print FPS values for both GPU and CPU processing.
   - It will display a bar chart comparing FPS.
   - It will also print a sample comparison of OCR results between GPU and CPU, including similarity scores.
## Script Overview
 - similarity(a, b): Calculates the similarity ratio between two text strings.
 - process_video(video_path, reader, device, frame_skip=5, resize_factor=0.5): Processes the video, performs OCR using the provided reader, and calculates the average FPS.
 - display_fps_comparison(gpu_fps, cpu_fps): Displays a bar chart comparing GPU and CPU FPS.
 - video_path: Path to the video file to process.
 - device_gpu: Sets the device for GPU or CPU processing.
 - reader_gpu and reader_cpu: EasyOCR readers for GPU and CPU.

# OpenCV Color Segmentation Object Tracking

Real-time object tracking project using HSV color segmentation, masking, and contour detection in OpenCV.

The project tracks a red ball in both a static image and a video sequence. It demonstrates how color thresholding can be used to isolate an object and follow its position frame by frame.

## Features

- Red object detection using HSV color space
- Binary mask generation
- Noise reduction with image filtering
- Contour detection
- Object localization in image and video
- Real-time video processing with OpenCV

## Technologies Used

- Python
- OpenCV
- NumPy

## Project Structure

```text
opencv-color-segmentation-object-tracking/
├── project1_photo.py
├── project1_video.py
├── red_ball.jpg
├── rgb_ball_720.mp4
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install opencv-python numpy
```

Run image detection:

```bash
python project1_photo.py
```

Run video tracking:

```bash
python project1_video.py
```

## Notes

This project was created as a computer vision practice task.

The main goal is to demonstrate basic object tracking using color segmentation and contour-based detection.

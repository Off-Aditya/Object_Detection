# Object Detection

This repository runs real-time object detection using OpenCV's DNN module with a pre-trained SSD MobileNet model.

## What it does

- Loads a TensorFlow SSD MobileNet object detection model from `model_data/`
- Reads class labels from `model_data/coco.names`
- Captures video from a camera or video file
- Detects objects in each frame
- Draws bounding boxes and labels on detected objects
- Counts objects per class per frame
- Speaks a summary of objects seen every 6 seconds using text-to-speech
- Displays the annotated video output in a window

## Files

- `main.py` - main entry point for running detection
- `Detector.py` - detector implementation using OpenCV DNN
- `sound.py` - text-to-speech support with `pyttsx3`
- `camera_test.py` - checks available camera indices on the machine
- `model_data/` - model files and COCO label names
- `test_videos/` - sample videos (if available)

## Requirements

- Python 3.8+ recommended
- `opencv-python`
- `numpy`
- `pyttsx3`

Install dependencies with:

```powershell
python -m pip install opencv-python numpy pyttsx3
```

## Running the detector

1. Open a terminal in the repository folder.
2. Activate your virtual environment if you have one.
3. Run:

```powershell
python main.py
```

By default, `main.py` uses `videoPath = 0`, which opens the first connected camera.

## Using a video file instead of the camera

Open `main.py` and change:

```python
videoPath = 0
```

to the path of a video file, for example:

```python
videoPath = r"test_videos\my_video.mp4"
```

## Controls

- Press `q` to quit the video window
- Press `p` to pause or resume detection

## Notes

- The model files are loaded from `model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` and `model_data/frozen_inference_graph.pb`.
- `Detector.py` uses a confidence threshold of `0.4` and non-maxima suppression to filter detections.
- Object summaries are spoken periodically using `sound.py`.

## Camera availability test

Run `camera_test.py` to print available camera indexes:

```powershell
python camera_test.py
```

If you want to change the speech frequency, update `speech_interval` in `Detector.py`.

# Dependencies
The application requires the following python dependencies:
- opencv-python
- ultralytics

To install the dependencies in the current python enviroment, run:
```
pip install opencv-python ultralytics
```

Alternatively, if you don't have a python enviroment setup, the `setup_env.sh` script will do it automatically and install the dependencies.
To run the script:
```shell
source setup_env.sh
```

# Running the application
For default functionality, run:

```python
python track_video.py 
```
This will run the satellite pose tracking program on the specified video `roe2_synthetic.mp4`.

# Arguments

Possible argument options:

- `--device`: Select what hardware to use for model inference. **Options**: `cpu`, `mps`, `cuda`
- `--video`: Select video to estimate the pose of. **Options**: `roe1_synthetic.mp4`, `roe2_synthetic.mp4`
- `--kps`: Provide the path of the selected keypoints of the object.
- `--no-save`: Don't save a video of the pose estimation.
- `--weights`: Provide the path to custom model weights.
- `--imgsz`: YOLO image size.
- `--brightness`: Brightness of each frame before inference.
- `--kp-conf`: Minimum keypoint confidence.
- `--no-show`: Headless mode.
- `--output`: Output video path.

# Example usage with arguments

```python
python track_video.py --video roe2_synthetic.mp4 --device mps --no-save --kps kps.npy
```

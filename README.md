# Crop-row Centerline Detection

Vision-based algorithm for crop-row centerline detection based on depth images (videos) captured by a ZED stereo camera.
The algorithm allows to calculate lateral and angular deviations from the center of detected line to the center of the camera.

Depth images are used for detection, while RGB images are used to plot the crop-row centerline and deviations.

Demo: [Video](https://youtu.be/ZX202kf5iLM)

### RGB Video

<img src="./demo/video_color.gif"/>

### Depth Video

<img src="./demo/video_depth.gif"/>

## Installation

Clone this repository.

```
git clone https://github.com/camilodelvalle/crop-centerline-detection
cd crop-centerline-detection
```

Create a virtual environment and activate it.

```
python3 -m venv .env
source .env/bin/activate
```

Install all dependencies required for the project.

```
pip3 install -r requirements.txt
```

## Usage

Run `centerline_detection.py` using the sample videos to generate the output, which will be saved in the **output_videos** folder.


```
cd crop_centerline_detection
python3 centerline_detection.py ../input_videos/video_rgb.mp4 ../input_videos/video_depth.mp4
```

### Output Video

<img src="./demo/video_output.gif"/>

## License

This project is licensed under the MIT License. See the LICENSE file.
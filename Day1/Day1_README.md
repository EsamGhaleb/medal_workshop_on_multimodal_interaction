# Repository Extracting MediaPipe Poses and Segmenting Gestures
# -*- coding: utf-8 -*-

## Gesture Segmentation Repository

This repository contains resources and scripts necessary segment gestures using skeletal models based on MediaPipe poses. It is part of the MEDAL workshop on multimodal interaction.

## Table of Contents
- 

## Getting Started
The requirements for this repository are in the parent directory: `../requirements.txt`. The setup instructions are also available in the parent directory: `../README.md`.


## Usage

### MediaPipe Pose Extraction
First, we will have a demonstration of how to use MediaPipe to extract poses from a video. This will be done using the `Mediapipe_Pose_Tutorial.ipynb` notebook. The notebook also contains instructions on how to run it as well as exercises to practice extracting poses from your own videos/webcam. 

### Gesture Segmentation
Next, we will segment gestures using the extracted subset poses. The segmentation is based on a model trained on the CABB dataset. The model is available in the `segmentation_models` directory.

The script `Gesture_Segmentation_Tutorial.ipynb` provides a demonstration of how to use the model to segment gestures from a video.  It has the following sections:

1. **Pose Extraction**: This section extracts poses from a video using MediaPipe. You can choose to use a video file or your webcam. The extracted poses are saved in a format suitable for the next step.

2. **Gesture Segmentation**: This section uses the pre-trained model to segment gestures from the extracted poses. 
   - The models are loaded from the `segmentation_models` directory.
   - The segmentation results are saved in a format that can be used for further analysis or visualization.

3. **Visualization**: The segmented gestures are visualized on the original video (with overlayed poses). This helps in understanding how well the model segments the gestures. The results are exported as Elan file.
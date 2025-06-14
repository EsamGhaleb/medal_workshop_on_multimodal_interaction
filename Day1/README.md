# Repository Extracting MediaPipe Poses and Segmenting Gestures
# -*- coding: utf-8 -*-

## Gesture Segmentation Repository

This repository contains resources and scripts necessary segment gestures using skeletal and speech models. It also generate ELAN files for the segmented gestures.

## Getting Started

These instructions will guide you to setup and run the project on your local machine.


### Generating ELAN Files

5. **Generate ELAN Files:**

   - After testing the model, you can generate ELAN files by executing:
     ```bash
     python save_segmented_gestures.py 
     ```
   - You can also check the segmentation examples in the following notebooks:

   ```bash
   jupyter notebook
   ```
   - Open the `notebooks/segment_gesture_data.ipynb` notebook to view the segmentation examples.

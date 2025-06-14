# Repository for speech and skeleton based gesture segmentation

## Gesture Segmentation Repository

This repository contains resources and scripts necessary segment gestures using skeletal and speech models. It also generate ELAN files for the segmented gestures.

## Getting Started

These instructions will guide you to setup and run the project on your local machine.

### Prerequisites

Setup virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install libsox-dev
```

### Pose Extraction and Sequential Data Generation for Sho Dataset

This section provides instructions on how to extract poses from a new dataset (e.g., new videos from Sho recordings) and generate sequential data.

#### Step 1: Extract Poses from Videos

To extract poses, follow the steps below:

1. Clone the **Gesture-Segmentation** repository and navigate to the `ViTPose` directory:

   ```bash
   cd ViTPose
   ```

   Please check the installation steps in the readme file in ViTPose
2. Run the following command to start extracting poses:

   ```srun
   srun python demo/landmark_detection_and_saving.py \
       demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
       https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
       https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
        --videos-dir /home/eghaleb/data \
       --output-dir ./selected_points \
   ```
3. **Important**: Replace the paths for:

   - `--videos-dir` with the directory containing your videos.
   - `--output-dir` with the directory where you want to save the extracted pose landmarks (preferable in data directory of the main project)

##### Note:

Extracting poses can take a lot of time, so it’s recommended to start the process as early as possible. 😅

#### Step 2: Extract Sequential Data

Once the pose extraction is complete, follow these steps to extract sequential data:

1. The script to extract sequential data from Sho videos is available in the repository. You can find it here:

   [generate_sequential_data_Sho.py](https://github.com/EsamGhaleb/Gesture-Segmentation/blob/main/data/generate_sequential_data_Sho.py)

#### Step 3: Running Inference to Extract Predictions

Once the data is ready, you can extract predictions using the inference script:

1. Run the inference script available here:

   [main_inference_alone.py](https://github.com/EsamGhaleb/Gesture-Segmentation/blob/main/main_inference_alone.py)
2. **Configuration**: Make sure to modify the config file to provide the correct data paths before running the inference script:

   [test_joints_sho.yaml](https://github.com/EsamGhaleb/Gesture-Segmentation/blob/main/config/detection/test_joints_sho.yaml)

That's it! Follow these steps carefully, and you’ll be able to extract poses, generate sequential data, and run inferences successfully. :star-struck:

#### Step 4: Generating ELAN Files
The final step is to generate ELAN files for the segmented gestures. You can find the script to generate ELAN files here:
   [save_segmented_gestures_sho.py](https://github.com/EsamGhaleb/Gesture-Segmentation/blob/main/save_segmented_gestures_sho.py)

The script will generate ELAN and save them in the `eaf_files_for_sho` directory.

### Downloading Data Pre Computed Poses

1. **Download Poses:**
   - Download the pose data from [Google Drive](https://drive.google.com/file/d/15SwxhEXC4JOJ0XYiQ-WcrGSmEVVvdIDB/view).
   - Make sure the downloaded files are extracted and placed in the `data/final_poses/` directory within this repository.

### Data Preparation

2. **Extract Poses:**
   - Run the following command to pre-process and prepare the poses from the data:
     ```bash
     python data/generate_sequential_data_large_cabb.py
     ```

### Model Training

3. **Train the Model:**

   - After preparing the data, you can train the model by executing:
     ```bash
     python main_detection.py --config config/detection/train_joints.yaml
     ```
   - To train the model with the speech features, execute:

   ```bash
   python main_detection.py --config config/detection/train_joints_speech.yaml
   ```

### Model Testing

4. **Test the Model:**

   - After training the model, you can generate test data by executing:
     ```bash
     python main_detection.py --config config/detection/train_joints_test.yaml
     ```
   - To test the model with the speech features, execute:

   ```bash
      python main_detection.py --config config/detection/train_joints_speech_test.yaml
   ```

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

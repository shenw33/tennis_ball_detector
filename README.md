# Tennis ball detection by OpenCV Cascade Classifier



## Data and Requirement

- Positive data

Here we use a demo video from @Tennibot.

The video file should be stored under:`data/raw_video/demo_video.avi`

The yolo annotation file should be stored under:`data/yolo_annotation/`

- Negative data
  Negative images should contain no tennis balls. As an example, the negative data is pre-downloaded. The source of the data is [ here ](https://www.kaggle.com/muhammadkhalid/negative-images)
  Save the data under: `data/negative_images/`

- Requirement lists

  The work is made possible with Python Scripts. The packages lists can be seen in "requirements.txt". It is deployed on Windows 10 and requires the pre-built software from OpenCV 3. These pre-built software should be saved under './data'

## Data Preparation

At this step, image frames from the raw video is collected under `data/raw_video/extracted_frames/`. Meanwhile, as the data received is annotation in the format of YOLO1.1, which is different from the requirement of OpenCV. It is also noticed that the annotation folder contains only the annotation label for the ones that contain at least one tennis ball. Therefore, effective training samples are collected and saved to a new folder "data/pos"

## Training

Training follows the config file stored under `data/config/`. The training is enabled by the pre-built software. You can build it by yourself or make use of the existing pre-built software in this folder. 

`train.py` creates the required samples and start training with the preset config file. Training results are stored in `./model`

## Evaluation

Since the time limitation and the fact that the OpenCV cascade classifier does not generate the confidence score for the inferencing, the single class object detection is not evaluated via mAP. 

The testing candidates by default are the extracted frames from the original video which contain both positive training data and negative frames.

Here in **evaluate.py**, object detection is inferenced and the maximum IoU values for each testing candidate is calculated and stored into yaml file.

## How to run

- Change directory into the source code

  ```bash
  cd tennis_ball_detector/src
  ```

- Prepare data by running

  ```bash
  python data_preparation.py
  ```

- To train the classifier:
  ```bash
  python train.py
  ```
- To simply evaluate the model
  ```bash
  python evaluate.py
  ```


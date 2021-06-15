import cv2
import os
import re
import pandas as pd
import numpy as np
import yaml
from ast import literal_eval



# os.chdir(os.path.dirname(__file__))  # switch the current working directory to the same directory as this script


def bb_iou(boxA, boxB):

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



def get_iou_max(box_label, box_pred):
    """
    Calculate the max iou for the bounding boxes for prediction and ground truth
    """
    common_boxes = min(len(box_label), len(box_pred))
    iou = []
    for k in range(common_boxes):
        iou.append(bb_iou(box_label[k], box_pred[k]))
    iou_max = 0 if len(iou) == 0 else max(iou)

    return iou_max


def validater(imagePath = '../data/raw_video/extracted_frames', cascPath = "../model/cascade.xml" ):	
    
    regex = r"frame_([0-9]*)" 
    df = pd.read_csv('truth_bboxes.csv')
    iou_table = []

    with open('../data/tennis.info') as f:
        lines = [line.rstrip() for line in f]
        count_label = 0
        # Loop for validating candidates
        for frame_count, img_file in enumerate(os.listdir(imagePath)):
            lines[count_label]
            tenniCascade = cv2.CascadeClassifier(cascPath)
            img_file = os.path.join(imagePath, img_file)
            image = cv2.imread(img_file)

            mysearch = re.search(regex, img_file)
            frame_file_name = 'frame_' + mysearch.group(1)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tennis_balls, _, weights = tenniCascade.detectMultiScale3(
                gray,
                scaleFactor =1.1,
                minNeighbors = 10,
                minSize = (10,10),
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True
            )

            print( "Found {0} tennis balls for the image file ".format( max(len(tennis_balls), 0 ) ) + frame_file_name)

            #Draw a rectangle around the detected objects
            box_pred = []
            box_label = []

            # Store the bbox information from groud truth
            if any(frame_file_name == df['frame']):
                box_label = df['BBox'][ int( np.where(frame_file_name == df['frame'])[0] ) ]
                box_label = literal_eval(box_label)

            for (x, y, w, h) in tennis_balls:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box_pred.append([x, y, x + w, y + h])
                
            box_label.sort()
            box_pred.sort()

            if len(box_label) * len(box_pred) == 0:
                # No prediction and also no ground truth
                iou_table.append(1)
            else:
                iou_table.append(get_iou_max(box_label, box_pred))
            # cv2.imwrite("saida.jpg", image)
            # cv2.imshow("validate_image", image)
            # cv2.waitKey(0)
    return iou_table

def save_yaml(iou_table, imagePath = '../data/raw_video/extracted_frames'):  

    test_imgs = os.listdir(imagePath)
    df= pd.DataFrame( columns = ['Test_ImageFrame', 'IoU_Max'])
    df['Test_ImageFrame'] = test_imgs
    df['IoU_Max'] = iou_table

    with open('../metrics/eval.yaml', 'w') as file:
        text = yaml.dump(
                    df.reset_index().to_dict(orient='records'),
                    sort_keys=False, width=72, indent=4,
                    default_flow_style=None)
        documents = yaml.dump(text, file)

if __name__ == "__main__":
    iou_table = validater()
    save_yaml(iou_table)






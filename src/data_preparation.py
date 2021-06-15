import cv2
import errno
import os
import re
import pandas as pd
# os.chdir(os.path.dirname(__file__))  # switch the current working directory to the same directory as this script

def make_new_dir(dirName):
    try:
        os.mkdir(dirName)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass  

def convert_entry(x, y, w, h, dw, dh):
    """
    Helper function to calculate the corresponding annotation information
    of a single entry

    x, y, w, h: the information from YOLO format
    dw, dh: image width and heigh
    
    Return:
    l, r, t, b: the returned results rectangle drawing
    """
    # convert them to opencv format
    l = max( int((x - w / 2) * dw), 0)
    r = min( int((x + w / 2) * dw), dw - 1)
    t = max( int((y - h / 2) * dh), 0)
    b = min( int((y + h / 2) * dh), dh - 1)  

    return l, r, t, b


def frame_extracter(video_path, frame_path):
    """
    To extract image frames from a video file and save them 
    under corresponding folder "extracted_frames/"

    @parameters:
    video_path: path to the video 
    frame_path: path to store the image frames from the video
    """
    # mkdir if frame folder not existed
    make_new_dir(frame_path)

    # Read Video and get the first frame
    vidcap = cv2.VideoCapture(video_path)
    isSuccess, image = vidcap.read()
    frame_count = 0
    # Extract all the frames and save them to folder
    print('Start reading and extracting image frames in the video ...')
    while isSuccess:
        cv2.imwrite(os.path.join(frame_path, "frame_{0:0=6d}.jpg".format(frame_count)), image)     # save frame as JPEG file      
        isSuccess, image = vidcap.read()
        # print('Read a new frame: ', isSuccess)
        frame_count += 1
    print("Successfully extracted {} frames!".format(frame_count))


def annotation_converter(label_path, image_path):
    """
    To convert the annotation in the format of YOLO to 
    openCV style with saving annotated image and useful image frames in a separate folder
    Create info file that contains the information for the annotation in the format of openCV

    label_path: the annotation files path
    image_path: the image frames to work with 

    """
    # image_path = '../data/extracted_images'
    number_files = len(os.listdir(label_path))
    print("There are {} annotation files in YOLO format".format(number_files), '\nConversion starts now ...')

    check_dir = os.path.join("../data/raw_video", 'annotated_check')
    img_dir = os.path.join("../data/", 'pos')

    make_new_dir(check_dir)
    make_new_dir(img_dir)
    # regular expression
    regex = r"frame_([0-9]*)"  
    # Create info data
    fname = os.path.join("../data/", "tennis.info")
    df = pd.DataFrame(columns = ['frame', 'BBox'])
    with open(fname, 'w') as f:
        # loop all the files in annotation folder  
        for frame_count, label_file in enumerate(os.listdir(label_path)):
            # Recognize the ID of frame in the annotation file via regular expression
            mysearch = re.search(regex, label_file)
            frame_file_name = 'frame_' + mysearch.group(1)

            # Read the corresponding image frame
            frame_img = cv2.imread(os.path.join(image_path, frame_file_name + '.jpg'))
            dh, dw, _ = frame_img.shape

            # Store the image data with annotation
            cv2.imwrite(os.path.join(img_dir, frame_file_name + '.jpg'), frame_img)
            # Read YOLO file
            yolo_file = open(os.path.join(label_path, frame_file_name + '.txt'), 'r')
            yolo_data = yolo_file.readlines()
            yolo_file.close()

            # To write basic info for this image file
            file_info_toWrite = 'pos/' + frame_file_name + '.jpg'
            f.write(file_info_toWrite)

            # loop all the annotation entries within an annotation file
            entries = ''
            entry_count = 0
            bboxes = []
            for dt in yolo_data:
                # split string to float
                _, x, y, w, h = map(float, dt.split(' ')) # Read YOLO format annotation
                left, right, top, bottom = convert_entry(x, y, w, h, dw, dh)

                # draw rect
                cv2.rectangle(frame_img, (left, top), (right, bottom), (0, 0, 255), 1)
                # save to opencv annotation data file
                box_w = right - left
                box_h = bottom - top
                entries += '{} {} {} {}  '.format(left, top, box_w, box_h)
                entry_count += 1
                bboxes.append([left, right, top, bottom])

            # Store the information for this annotation file
            df.loc[frame_count] = [frame_file_name, bboxes]
            f.write(' {} '.format(entry_count) + entries + '\n')
            
            cv2.imwrite(os.path.join(check_dir, "labeled_frame_{0:0=6d}.jpg".format(frame_count)), frame_img)
             
        print("Successfully converted annotation format for {} frames!".format(frame_count + 1))
    f.close()
    df.to_csv('truth_bboxes.csv', index=False)

def negative_processor(neg_path = '../data/negative_images'):
    """
    To load negative dataset and save the bg.txt with resizing 
    negative dataset from https://www.kaggle.com/muhammadkhalid/negative-images

    """
    fname = os.path.join("../data/", "bg.txt")
    img_dir = os.path.join("../data/", 'neg')

    make_new_dir(img_dir)

    with open(fname, 'w') as f:
        # loop all the files in annotation folder  
        for img_count, img_name in enumerate(os.listdir(neg_path)):
            # Read the corresponding image frame
            neg_img_path = os.path.join(neg_path, img_name)
            neg_img = cv2.imread(neg_img_path)
            resized = cv2.resize(neg_img, (200, 200), interpolation = cv2.INTER_AREA)  
            cv2.imwrite(os.path.join(img_dir, "neg_{0:0=6d}.jpg".format(img_count)), resized)
            
            f.write(os.path.join("neg", "neg_{0:0=6d}.jpg".format(img_count) + '\n')) 
        print("Successfully read and load {} negative images!".format(img_count + 1))
   
    f.close()

    pass



if __name__ == "__main__":

    positive_video_path = "../data/raw_video/demo_video.avi"
    positive_frame_path = "../data/raw_video/extracted_frames/"
    label_path = '../data/yolo_annotation/'
    frame_extracter(video_path = positive_video_path, frame_path = positive_frame_path)
    annotation_converter(label_path = label_path, image_path = positive_frame_path)
    negative_processor()
import subprocess
import json
import os
import errno

# os.chdir(os.path.dirname(__file__))  # switch the current working directory to the same directory as this script

def cascade_trainer():
    # Opening JSON file
    f = open('../config/config.json',)
    # returns JSON object as a dictionary
    config_data = json.load(f)
    f.close()

    numPos, numNeg = config_data['numPos'], config_data['numNeg']
    numStages = config_data['numStages']
    featureType = config_data['featureType']
    minHitRate = config_data['minHitRate']

    # subprocess.call("cd ../data", shell=True)
    os.chdir("../data")

    create_cmd = "opencv_createsamples -info tennis.info -num 534 -w 20 -h 20 -vec tennis.vec"
    rc = subprocess.call(create_cmd, shell=True)


    train_cmd = "opencv_traincascade -data ../model -vec tennis.vec -bg bg.txt -numPos {} -numNeg {} -numStages {} -minHitRate {}".format(numPos, numNeg, numStages, minHitRate)

    train_cmd += " featureType " + featureType
    train_cmd += " -w 20 -h 20"

    rc = subprocess.call(train_cmd, shell=True)

if __name__ == "__main__":
    cascade_trainer()
# NEGATIVE IMAGE FROM
# https://www.kaggle.com/muhammadkhalid/negative-images




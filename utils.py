import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_classification_data(images_dir='trodo-v01/images', target_path='trodo-v01/ground-truth/groundtruth.json', image_size=(256, 256), test_size=0.1):
    LABEL_MAP = {'analog': 0, 'digital': 1}

    with open(target_path) as file:
        data = json.load(file)['odometers']

    X = np.zeros((len(data), *image_size, 3), dtype=np.float32)
    Y = np.zeros((len(data), 1), dtype=np.float32)
    
    for i in range(len(data)):
        image_path = os.path.join(images_dir, data[i]['image'])
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA) / 255.0

        X[i] = image
        Y[i] = LABEL_MAP[data[i]['odometer_type']]

    X = np.reshape(X, (len(data), image_size[0] * image_size[1] * 3))

    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_size, random_state=42)
    return trainX, trainY, testX, testY

def json2txt(data, output):
    """
    Converts json file's input data (image_path and odometr_type) to txt to use in classification task.

    :param json_file: the path of json file
    :param output: the output path of txt file
    """

    mapping = {'analog': 0, 'digital': 1}

    with open(output, 'w') as out:
        for i in range(len(data)):
            out.write(f'{data[i]["image"]},{mapping[data[i]["odometer_type"]]}\n')
    


    



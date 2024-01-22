import os
import cv2
import json
import shutil
import pathlib
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def get_coords(data, annotation_dir):
    file_name = data['image'].split(".")[0]
    annotation_name = file_name + '.xml'
    annotation_path = os.path.join(annotation_dir, annotation_name)

    tree = ET.parse(annotation_path)
    root = tree.getroot()  
    width, height = int(root.find('size')[0].text), int(root.find('size')[1].text)

    for object in root.findall('object'):
        name = object.find('name')
        if name.text == 'odometer':
            bbox = object.find('bndbox')
            x1, y1 = float(bbox[0].text), float(bbox[1].text)
            x2, y2 = float(bbox[2].text), float(bbox[3].text)
            w, h = x2 - x1, y2 - y1
            break

    x = x1 / width
    y = y1 / height
    w = w / width
    h = h / height 

    return x, y, w, h, file_name


def main():
    LABEL_MAP = {"analog": 0, "digital": 1}
    IMAGES_DIR = "trodo-v01/images"
    ANNOTATION_DIR = "trodo-v01/pascal voc 1.1/Annotations"
    GROUND_TRUTH_PATH = "trodo-v01/ground-truth/groundtruth.json"
    OUTPUT_PATH = 'dataset' 

    pathlib.Path(os.path.join(OUTPUT_PATH, 'train')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(OUTPUT_PATH, 'val')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(OUTPUT_PATH, 'test')).mkdir(parents=True, exist_ok=True)

    with open(GROUND_TRUTH_PATH) as file:
        data = json.load(file)['odometers']

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    test, val = train_test_split(test, test_size=0.5, random_state=42)


    for split in [(train, 'train'), (val, 'val'), (test, 'test')]:
        dataset, split_name = split[0], split[1]

        for i in range(len(split)):
            image_path = os.path.join(IMAGES_DIR, dataset[i]['image'])
            odometer_type = LABEL_MAP[dataset[i]['odometer_type']]
            x, y, w, h, file_name = get_coords(dataset[i], ANNOTATION_DIR)

        with open(f'{OUTPUT_PATH}/{split_name}/{file_name}.txt', 'w') as f:
            f.write(f'{odometer_type} {x} {y} {w} {h}\n')
            shutil.copy(image_path, f'{OUTPUT_PATH}/{split_name}/{dataset[i]["image"]}')

if __name__ == '__main__':
    main()

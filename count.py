import json
import os
import xml.etree.ElementTree as ET

LABEL_MAP = {"analog": 0, "digital": 1}


annotation_dir = "trodo-v01/pascal voc 1.1/Annotations"
images_dir = "trodo-v01/images"
ground_truth_path = "trodo-v01/ground-truth/groundtruth.json"
output = 'dataset' 

counter = {}


with open(ground_truth_path) as file:
    data = json.load(file)['odometers']


for i in range(len(data)):
    file_name = data[i]['image'].split(".")[0]
    annotation_name = file_name + '.xml'
    annotation_path = os.path.join(annotation_dir, annotation_name)     

    tree = ET.parse(annotation_path)
    root = tree.getroot()  
    width, height = int(root.find('size')[0].text), int(root.find('size')[1].text) 

    key = f'{width}x{height}'
    if key not in counter:     
        counter[key] = 0
    else:
        counter[key]+=1

print(counter)          

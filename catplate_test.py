import os
from datetime import datetime
import uuid 
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def savefImg(file):
    now = datetime.now()
    date = now.strftime('%d%m%Y_%H%M%S')
    unique_id = os.path.join('Saved detection', 'plate_img_{}'.format(date))

    target_dir = 'Saved detection'

    file_path = unique_id + '.jpeg'
    cv2.imwrite(file_path, file)


    


#load YOLOv3 and YOLOv3 custom
net = cv2.dnn.readNet('weights/yolov3_plate.weights', 
                      'YoloV3 cfg/yolov3_plate.cfg')
net_car = cv2.dnn.readNet('weights/yolov3_land_vehicle.weights', 'YoloV3 cfg/yolov3_land_vehicle.cfg')

font = cv2.FONT_HERSHEY_DUPLEX
colors = np.random.uniform(0, 255, size=(100, 3))
text = "Press 'q' to quit"

boxes = []
confidences = []
class_ids = []

img = cv2.imread(r"C:/Users/sobsd/OneDrive/Pictures/dataset-card.jpg")
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_img = img[y:y+h, x:x+w]
        savefImg(crop_img)
           

fig = plt.figure(figsize=(10,10))
plt.imshow(crop_img)
plt.show()


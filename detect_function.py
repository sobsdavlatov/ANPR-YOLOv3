
import os
from datetime import datetime

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#load YOLOv3 and YOLOv3 custom
net = cv2.dnn.readNet('Weights/yolov3_plate.weights', 
                      'YoloV3 cfg/yolov3_plate.cfg')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#easyOCR pipeline
reader = easyocr.Reader(['en'], gpu=True)

font = cv2.FONT_HERSHEY_DUPLEX
colors = np.random.uniform(0, 255, size=(100, 3))
text = "Press 'q' to quit"

boxes = []
confidences = []
class_ids = []

#fucntion to detect plate
def detectPlate(file):
    img = cv2.imread(file)
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
            if confidence > 0.8:
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
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
            crop_img = img[y:y+h, x:x+w]
            saved = savefImg(crop_img)
            return saved

#function to save files
def savefImg(file):
    now = datetime.now()
    date = now.strftime('%d%m%Y_%H%M%S')
    unique_id = os.path.join('Saved detection', 'plate_img_{}'.format(date))

    target_dir = 'Saved detection'

    file_path = unique_id + '.jpeg'
    cv2.imwrite(file_path, file)
    return file_path

#function to read text from plate
def readPlate(file):
    result = reader.readtext(file)
    result 
    img = cv2.imread(file)
    for detection in result:
        text = detection[1]
        #saving the result into csv file
        with open('Saved detection/numbers.csv', 'a') as n:
            n.write(text + '\n')
    return text

def detectFromImg(file):
    plate_img = detectPlate(file)
    plate_number = readPlate(plate_img)

    plate_imgPLT = plt.imread(plate_img)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(plate_imgPLT)
    ax.text(0, -10, plate_number, fontsize=12, color='white',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
    plt.show()
    
def detectFromVid(file):
    cap = cv2.VideoCapture(file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))


    while cap.isOpened():
        ret, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.8:
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
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    crop_img = img[y:y+h, x:x+w]
        
            out.write(img)
            cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




                    



    

    

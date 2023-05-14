from detect_function import detectFromVid
import numpy as np
import cv2
net = cv2.dnn.readNet('Weights/yolov3_plate.weights', 
                      'YoloV3 cfg/yolov3_plate.cfg')

cap = cv2.VideoCapture(r'C:/Users/sobsd/Downloads/Y2Mate.is - Porsche 992 GT3 Touring  Night Vibes 4K-da5x-__kvGY-720p-1654210013133.mp4')

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

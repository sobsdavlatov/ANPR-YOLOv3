# ANPR
The ANPR (Automatic Number Plate Recognition) system that utilizes a custom YOLO-v3 model and optical character recognition is a powerful tool designed to detect land vehicles and their registration plates. The system uses a combination of deep learning and computer vision techniques to identify vehicles in real-time, capture their registration plates, and save the information into a CSV file.

Once the number plates are detected, the optical character recognition (OCR) component of the system kicks in. OCR is a process that uses computer vision algorithms to recognize and extract text from images. The ANPR system applies OCR to the number plates to extract the alphanumeric characters and convert them into machine-readable text.
![Figure_1](https://github.com/sobsdavlatov/ANPR-YOLOv3/assets/132314169/21736c46-12e5-4377-96d8-77b77e1cc9d3)
The ANPR system then saves the extracted data into a CSV file, which can be used for further analysis or integration with other systems. The CSV file contains a timestamp of when the vehicle was detected, the registration plate number, and the location of the vehicle.
### The program also detects from video input.
https://github.com/sobsdavlatov/ANPR-YOLOv3/assets/132314169/983dd9fb-a617-4169-83bf-7b7842b4cd4d

# How to run the program
1. To use system donwload Weights from this link https://drive.google.com/drive/u/0/folders/1h7R4GFkNrsl3GnqQ0D83MhMiz8pvFGFE Create "Weights" folder inside of directory and put these files inside the folder. 
2. Run 'main.py' script
# The program works with .jpeg and .mp4 files. The program aslo has a feature to detect from camera. 

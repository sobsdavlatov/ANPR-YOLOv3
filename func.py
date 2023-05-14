import tkinter as tk 
from tkinter import filedialog
import matplotlib.pyplot as plt
from detect_function import  detectFromImg, detectFromVid


#function for main window
def mainwin():
    global root 
    root = tk.Tk()
    root.geometry('200x200')
    root.title("ANPR")


#function to call filedialog
def callback():
    root.filename = filedialog.askopenfilename(parent=root, 
                                               initialdir= "/path/to/start",
                                               title = "Choose a file")
    filetype(root.filename)

#function to check filetype
def filetype(file):
    if file.endswith('.jpg' or '.jp2' or '.png'):
        detectFromImg(file)
    else: 
        detectFromVid(file)





























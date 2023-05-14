from logging import root
import tkinter as tk
from unittest import TextTestResult
import func as fc 
import detect_function as df
import cv2
#initialize the main window 
fc.mainwin()
#buttons 
filedialogButton = tk.Button(fc.root, text='Choose file', command= lambda:fc.callback()) #all together function should be passed 
filedialogButton.pack()

if __name__ == "__main__":
    tk.mainloop()
    



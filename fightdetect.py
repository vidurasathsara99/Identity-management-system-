import imutils
import pickle
import time
import cv2
import os
from imutils import paths
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import shutil
from pathlib import Path
import os
import datetime

class MyWindow:
    def __init__(self, win):
       
        self.lbl3 = Label(win, text='Video Path :')
       
        self.t3 = Entry()
        self.btn1 = Button(win, text='Model Train')
        self.btn2 = Button(win, text='Detect Person')
        self.b2 = Button(win, text='Detect Persons', command=self.train_)
        self.b1 = Button(win, text='Add Video', command=self.select_file_vid)
        self.person = ''
        self.b1.place(x=100, y=250)
        self.b2.place(x=200, y=250)  
        self.lbl3.place(x=100, y=215)
        self.t3.place(x=200, y=215)

  

    def select_file_vid(self):
        filetypes = (
            ('image files', '*.mp4'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a Video file',
            initialdir='/',
            filetypes=filetypes)
        self.t3.insert(END, str(filename))



    def train_(self):
        
        print("Streaming started")
        video_capture = cv2.VideoCapture(self.t3.get())
        print(self.t3.get().rsplit('/', 1)[-1])
        predict=self.t3.get().rsplit('/', 1)[-1]
        prediction = predict[0]
        print (prediction)
        # loop over frames from the video file stream
        while True:
            success,img = video_capture.read()
            if prediction=="f":
                text ="Alert: Fighting "
                cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,
                              3,(255,0,0),3)
                cv2.imshow('image',img)
                cv2.waitKey(10)
            else:
                text="Alert: Normal "
                cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,
                              3,(255,0,0),3)
                cv2.imshow('image',img)
                cv2.waitKey(10)
        video_capture.release()
        cv2.destroyAllWindows()



window = Tk()
mywin = MyWindow(window)
window.title('Person Detector')
window.geometry("500x400+20+20")
window.mainloop()

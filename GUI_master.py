
import tkinter as tk
from tkinter import ttk, LEFT, END
import time
import numpy as np
import cv2
import os
from PIL import Image , ImageTk     
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 

##############################################+=============================================================

root = tk.Tk()
#root.configure(background="seashell2")
#root.geometry("1300x700")
import sqlite3
my_conn = sqlite3.connect('face.db')

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Driver Drowsiness ")


#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('cd.jpg')
image2 =image2.resize((w,h), Image.NEAREST)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)



  #################################################################################################################
def reg():
    print("Registration")
    from subprocess import call
    call(["python", "registration.py"]) 



def Log():
    print("Login")
    from subprocess import call
    call(["python", "login.py"])
 
def mode():
    print("model")
    from subprocess import call
    call(["python", "model.py"])


def window():
    root.destroy()


button1 = tk.Button(root, text="Registration", command=reg,width=20, height=1, font=('times', 15, ' bold '),bg="brown",fg="white")
button1.place(x=200, y=410)

button2 = tk.Button(root, text="Login", command=Log,width=20, height=1, font=('times', 15, ' bold '),bg="green",fg="white")
button2.place(x=600, y=410)


##
#
#button5 = tk.Button(frame_alpr, text="button5", command=window,width=20, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
#button5.place(x=10, y=280)


exit = tk.Button(root, text="Exit", command=window, width=20, height=1, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=1000, y=410)



root.mainloop()
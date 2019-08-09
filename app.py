from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import sys# impo\srt the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk

import tkinter.filedialog
import cv2
from keras.models import model_from_json
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
resImage=""
def resize( image, width=None, height=None, inter=cv2.INTER_AREA):

	dim = (width, height)

	resized = cv2.resize(image, dim, interpolation=inter)

	return resized
def saveimage():
	cv2.imwrite("C:/Users/Leo/Downloads/asdf.jpg",resImage)
def select_image():
	# grab a reference to the image panels
	global panelA, panelB, resImage
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path =  tkinter.filedialog.askopenfilename()
	print(path)
    # ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# edged = cv2.Canny(gray, 50, 100)
 
		# # OpenCV represents images in BGR order; however PIL represents
		# # images in RGB order, so we need to swap the channels
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		colorize=[]
		# img=img_to_array(load_img(path))
		image=resize(image,width=256,height=256)
		print(image.shape)
		colorize.append(image)
		colorize = np.array(colorize, dtype=np.float64)
		colorize = rgb2lab(1.0/255*colorize)[:,:,:,0]
		colorize = colorize.reshape(colorize.shape+(1,))
		
		
		# Test model
		output = loaded_model.predict(colorize)
		output = output * 255
        # Output colorizations
		for i in range(len(output)):
			cur = np.zeros(image.shape)
			cur[:,:,0] = colorize[i][:,:,0]
			cur[:,:,1:] = output[i]
			resImage = lab2rgb(cur)
		max=np.max(resImage)
		resImage=	resImage / max
		resImage=	resImage *255
		# resImage = 255 * 
		resImage=resImage.astype(np.uint8)
		print(image.dtype)
		print(image.shape)
		print(type(image))
		print(type(resImage))
		# convert the images to PIL format...
		image = Image.fromarray(image)
		resImg= Image.fromarray(resImage)
		resImg.save('C:/Users/Leo/Downloads/asdf.jpg')
		# ...and then to ImageTk format
		image = ImageTk.PhotoImage(image)
		resImg = ImageTk.PhotoImage(resImg)
        # if the panels are None, initialize them
		if panelA is None or panelB is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
 
			# while the second panel will store the edge map
			panelB = Label(image=resImg)
			panelB.image = resImg
			panelB.pack(side="right", padx=10, pady=10)
 
		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image)
			panelB.configure(image=resImg)
			panelA.image = image
			panelB.image = resImg    

# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
 
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
# kick off the GUI
root.mainloop()


    

        
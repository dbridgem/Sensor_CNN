# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:12:01 2019

This is a convolutional neural net for production quality control

@author: devon
"""

import numpy as np
import pandas as pd
import os
import PIL
from PIL import ImageTk,Image,ImageChops
from tkinter import Tk,Canvas,Label,Button,Toplevel
import tkinter as tk
from tkinter import filedialog
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation

#First I have it to get all filenames from a specified folder in the directory.
#In this case it is 'Plate Images_3-28-19'
def listgrab(folder):
    os.chdir(folder)
    filelist = os.listdir()
    os.chdir('..')
    num_files = len(filelist)
    i = 0
    for files in filelist:
        filelist[i] = folder + '/' + filelist[i]
        i += 1
        
    return(filelist,num_files)
    
#Since the orientation was not reliable in the setup, this is a quick check to ensure
#that the images are in the same orientation base on a reference image    
def imflip(filelist):
    
    imageref = Image.open('Image_positionref.JPG')
    [width,height] = imageref.size
    width = int(width / 128)
    height = int(height / 128)
    imageref = imageref.resize((width,height),Image.ANTIALIAS)
    for file in filelist:
        image = Image.open(file)
        tempim = image.resize((width,height),Image.ANTIALIAS)
        
        flipped_imagedif_0 = ImageChops.subtract(tempim,imageref,scale = 1,offset = 0)
        flipped_imagedif_90 = ImageChops.subtract(tempim.rotate(90),imageref,scale = 1,offset = 0)
        flipped_imagedif_180 = ImageChops.subtract(tempim.rotate(180),imageref,scale = 1,offset = 0)
        flipped_imagedif_270 = ImageChops.subtract(tempim.rotate(270),imageref,scale = 1,offset = 0)
        
        flipped_imagedif_0 = np.asarray(flipped_imagedif_0, dtype=np.float32)
        flipped_imagedif_90 = np.asarray(flipped_imagedif_90, dtype=np.float32)
        flipped_imagedif_180 = np.asarray(flipped_imagedif_180, dtype=np.float32)
        flipped_imagedif_270 = np.asarray(flipped_imagedif_270, dtype=np.float32)
        
        flipped_imagedif_0 = np.mean(np.mean(np.mean(flipped_imagedif_0)))
        flipped_imagedif_90 = np.mean(np.mean(np.mean(flipped_imagedif_90)))
        flipped_imagedif_180 = np.mean(np.mean(np.mean(flipped_imagedif_180)))
        flipped_imagedif_270 = np.mean(np.mean(np.mean(flipped_imagedif_270)))
        
        mindiff = [flipped_imagedif_0,flipped_imagedif_90,flipped_imagedif_180,flipped_imagedif_270]
        mindiff = mindiff.index(min(mindiff))
        
        if mindiff == 1:
            image = image.rotate(90)
        elif mindiff == 2:
            image = image.rotate(180)
        elif mindiff == 3:
            image = image.rotate(270)    
            
        image.save(file)
 
#This is to take the images, divide them into 16 parts (1 per individual sensor on the plate),
#then create and save an image stack that can be fed directly into the CNN
def imageslicer(filelist):
    
    sensordata = np.empty([len(filelist)*16,400,275,3])
    i = 0
    for file in filelist:
        fullimage = Image.open(file)
        leftbounds = [475,775,1090,1385,1680,1980,2280,2600,
                      500,800,1100,1400,1700,2000,2300,2600]
        topbounds = [200,200,200,200,190,190,180,170,
                     970,970,950,950,940,940,935,920]
    
        sensorareaimage = np.zeros([16,400,275,3])
        for j in range(16):#iterate for each sensor area (1 for now)
            sensorarea = (leftbounds[j],topbounds[j],leftbounds[j]+275,topbounds[j]+400)
            sensorareaimage[j] = fullimage.crop(sensorarea)
            
        sensordata[i:i+16,:,:,:] = sensorareaimage        
        i += 16
    
    #currentdata = np.load(open("CNNSensorsData.csv", "rb"), delimiter=",", skiprows=0)
    #currentdata = np.concatenate(currentdata,sensordata,axis=0,out=None)
    np.save("CNNSensorsData", sensordata)
    return()

#This is a little UI to classify the images for training and validation.
#It will iterate through the saved image stack, let the user give a pass or fail
#score to the sensor, then save the results in a format that the CNN can use.
def manualclassification():
    
    def passcondition():
        global picindex
        global imclass
        imclass[picindex] = 0
        picindex += 1
        getimage()
        print(picindex)
        
    def failcondition():
        global picindex
        global imclass
        imclass[picindex] = 1
        picindex += 1
        getimage()      
        print(picindex)
        
    def getimage():
        try:
            sensorimage = Image.fromarray(np.uint8((256-imagestack[picindex,:,:,:])))
            photoimg = ImageTk.PhotoImage(sensorimage)
            panel.configure(image=photoimg)
            panel.image = photoimg
        except:
            imclass.to_csv('classes.csv',header = None, index = False)
        
    imagestack = np.load('CNNSensorsData.npy')
    [m,n,o,p] = imagestack.shape
    
    global picindex
    global imclass
    imclass = np.zeros(m)
    picindex = 0
    
    
    root=Tk()
    
    sensorimage = Image.fromarray(np.uint8((256-imagestack[picindex,:,:,:])))
    photoimg = ImageTk.PhotoImage(sensorimage)
    panel = Label(root,image = photoimg)
    panel.grid(rowspan = 4, columnspan = 4,row=0,column=0)
    
    passbtn = Button(root, text="Pass", command = passcondition)
    passbtn.grid(row=4,column=1)
    failbtn = Button(root, text="Fail", command =failcondition)
    failbtn.grid(row=4,column=3)
    
    #canvas.create_image(0,0, image = img)
    
    root.mainloop()

#This will split the data randomly into a train and test set
def CNNsplit():
    imagestack = np.load('CNNSensorsData.npy')
    classes = pd.read_csv('classes.csv')
    
    a = classes
    b = imagestack[:-1,50,50,0]
    randomize = np.arange(len(classes))
    np.random.shuffle(randomize)
    classes = classes.iloc[randomize]
    classes = classes.reset_index(drop=True)
    imagestack = imagestack[randomize,:,:,:]

    ar = classes
    br = imagestack[:,50,50,0]

    mtest = int(np.floor(len(imagestack)/10))
    mtrain = len(imagestack) - mtest
    imstack_test = imagestack[:mtest,:,:,:]
    imstack_train = imagestack[-mtrain:,:,:,:]
    classes_test = classes.iloc[:mtest]
    classes_train = classes.iloc[-mtrain:]    
    np.savez_compressed('imstack_test', imstack_test)
    np.savez_compressed('imstack_train', imstack_train)
    classes_test.to_csv('classes_test.csv',header=False,index=False)
    classes_train.to_csv('classes_train.csv',header=False,index=False)
    
#This is where the CNN actually is. I take the images, downsample them to avoid 
#melting my ram (by half in this case), then feed them into the CNN for training.
#Keras then trains the model and gives an assessment of the accuracy
def CNNtrain():
    batch_size = 128
    num_classes = 2
    epochs = 12
    
    # input image dimensions
    img_rows, img_cols = 200, 138
    
    # the data, split between train and test sets
    x_train = np.load('imstack_train.npz')
    x_train = x_train.f.arr_0
    x_train = x_train[1:]
    x_train = x_train[:,::2,::2,:]
    y_train = pd.read_csv('classes_train.csv')
    x_test = np.load('imstack_test.npz')
    x_test = x_test.f.arr_0
    x_test = x_test[1:]
    x_test = x_test[:,::2,::2,:]
    y_test = pd.read_csv('classes_test.csv')
    
    #x_train = x_train.reshape(375, 400, 275, 3)
    #x_test = x_test.reshape(374, 400, 275, 3)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(100*2, 69*2, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) 
        

    
    
if __name__ == '__main__':   
    
    (filelist,num_files) = listgrab('Plate Images_3-28-19')#Use to process the files in "Loose Files" folder
    
    imflip(filelist)#Put images in the right orientation

    imageslicer(filelist) #Extract closeup images of sensors from the filelist grabbed above

    manualclassification() #Widget for user to classify images as pass or fail

    CNNsplit() #Split the data into the test and train sets

    CNNtrain()
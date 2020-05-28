# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:45:19 2020

@author: nagar
"""

import os
os.getcwd()
os.chdir('D:\HAP880\Project')
import numpy as np
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt

import cv2


df=pd.read_csv('Data_Entry_2017.csv')
df.columns


newdf=df[df['Finding Labels']=='Effusion']
newdf=newdf.reset_index()
newdf1=newdf['Patient ID'].value_counts()
newdf1=newdf1.reset_index()

        
final=[]
#y=[]
for i in range(1,13):
    if i <10:
        pp='D:\HAP880\Project\data\images_00'+str(i)+'\images'
    else:
       pp='D:\HAP880\Project\data\images_0'+str(i)+'\images' 
    os.chdir(pp)
    for i in range(len(newdf1)):
        p_id=newdf1['index'][i]
        temp=newdf[newdf['Patient ID']==p_id]
        fup_max=temp[temp['Follow-up #']==max(temp['Follow-up #'])]
        fup_max=fup_max.reset_index()                  
        filename = newdf['Image Index'][i]
        if os.path.exists(filename):
            img = cv2.imread(filename)
            x=cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('resized',x)
            final.append(x)
            #y.append(4)
    

len(final)
final=np.array(final)
os.chdir('C:\Spring 2020\HAP 880\Project')
np.save('Atelectasis',final)
#np.save('all_y',y1)

final[0].shape

##############################################################################
## Combine two diseases for testing
import os
os.getcwd()
os.chdir('C:\Spring 2020\HAP 880\Project')
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random

at=np.load('Atelectasis.npy')
nf=np.load('Infiltration.npy')

at=at/255
nf=nf/255
at=at.astype('float32')
nf=nf.astype('float32')


label1=np.ones(len(at))
label2=np.zeros(len(nf))
len(at)

final_x=np.concatenate((at,nf))
final_y=np.concatenate((label1,label2))

l=[]
for i in final_y:
    if i==1.:
        l.append(np.array([0.,1.]))
    else:
        l.append(np.array([1.,0.]))
        
final=np.array(l)

np.save('Atelectasis_Infiltration',final_x)
np.save('Atelectasis_Infiltration_y',final)

len(at)
len(nf)
###############################################

X=np.load('Atelectasis_Infiltration.npy')
Y=np.load('Atelectasis_Infiltration_y.npy')
len(Y)
#generate a balanced set for testing
X=X[0:5300]
Y=Y[0:5300]


len(X)


np.save('Atelectasis_Infiltration_X',X)
np.save('Atelectasis_Infiltration_y',Y)

###############################################################
# Visualize some images
X=np.load('Atelectasis_Infiltration_X.npy')
Y=np.load('Atelectasis_Infiltration_y.npy')
X.shape
cols = 4
rows = 2

fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        k=random.randint(0,5300)
        if Y[k][0]==0. and Y[k][1]==1.:
            x='Atelectasis'
        else:
            x='Infiltration'
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.set_title(x)
        ax.imshow(X[k, :])
        
plt.show()

##############################################################################

from numpy import expand_dims

from matplotlib import pyplot
import cv2

# convert to numpy array
data = X[15]
type(data)
cv2.imshow('resized',data)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(samplewise_center=True,
                  samplewise_std_normalization=True,
                  horizontal_flip=True,
                  vertical_flip=False,
                  height_shift_range=0.05,
                  width_shift_range=0.1,
                  rotation_range=5,
                  shear_range=0.1,
                  fill_mode='reflect',
                  zoom_range=0.15)
# prepare iterator
it = datagen.flow(samples,batch_size=10)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('float32')
	# plot raw pixel data
	pyplot.imshow(image[:,:,0],cmap='bone')
# show the figure
pyplot.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:06:57 2020

@author: zahid
"""
#%% libraries

import tensorflow as tf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import matplotlib.pyplot as plt

import numpy as np

import cv2

import glob

from scipy.io import loadmat

import random
from random import seed, randint

from sklearn.model_selection import train_test_split
#%%  Data Load Parts

# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'

#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'


dataPath = os.path.join(iD_ir, '*.pgm')
files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# load images  from 1 subject
#%% Load Video

data = []
im_size = (100,100)

i = 0
for f1 in files:  # make sure to stack serially
    if i%1000==0:
        print(f1)
    img =  cv2.resize(cv2.imread(f1)[:,:,1], im_size)
    img = img[:,:,np.newaxis]
    data.append(img)
    i+=1
    
data =  np.array(data)
    
#%% load Mat file
x = loadmat('../../../Dataset/Merl_Tim/Subject1_still/PulseOX/pulseOx.mat')

pulseoxR = np.squeeze(x['pulseOxRecord'])

pulR = []
for i in range(pulseoxR.shape[0]):
    pulR.append(pulseoxR[i][0][0])
    
pulR = np.array(pulR)
    
#%% Prepare dataset for training
random.seed(1)

rv = [randint(0,5300) for _ in range(5000)]
randint(0,5350)
rv =  np.array(rv)
pulR = np.reshape(pulR, [10703,1])
#      #%%

trainX = []
#trainY = np.zeros([200,80])
trainY = []

frame_cons = 40 # how many frame to consider at a time

for j,i in enumerate(rv):
    img = np.reshape(data[i:i+frame_cons,:,:,0], [frame_cons,*im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    ppg = pulR[2*i: 2*i+80,0]
    trainY.append(ppg)


trainX = np.array(trainX, dtype = np.float32)
trainY = np.array(trainY, dtype = np.float32)

#%% Some parameter definition

num_classes = 80 # total classes (0-9 digits).
num_features = 100*100*40 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.001 # start with 0.001
training_steps = 30000
batch_size = 8
display_step = 100


# Network parameters.


#%%    Normalize and split

trainX = (trainX-trainX.min())/(trainX.max()-trainX.min())
trainY = (trainY-trainY.min())/(trainY.max()-trainY.min())

trX, teX, trY, teY = train_test_split(trainX , trainY, test_size = .1, random_state = 42)

#%% tensorflow dataload

train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
train_data = train_data.repeat().shuffle(buffer_size=5, seed= 1).batch(batch_size).prefetch(1)



#%% Loss function  

def RootMeanSquareLoss(x,y):
    loss = tf.keras.losses.MSE(y_true = y, y_pred =x)
    #return tf.reduce_mean(loss) 
    return loss

#%%  Optimizer Definition
optimizer = tf.optimizers.SGD(learning_rate)

def run_optimization(neural_net, x,y):    
    with tf.GradientTape() as g:
        pred =  neural_net(x, training = True)
        loss =  RootMeanSquareLoss(y, pred)
        
    trainable_variables =  neural_net.trainable_variables
    
    gradients =  g.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
def train_nn(neural_net, train_data):
        
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):     
        run_optimization(neural_net, batch_x, batch_y)

        if step % display_step == 0:
            pred = neural_net(batch_x, training=True)
            loss = RootMeanSquareLoss(batch_y, pred)
            print("step: %i, loss: %f" % (step, tf.reduce_mean(loss)))
      

#%% Bringing Network
from net_work_def import ConvNet
neural_net = ConvNet(num_classes)


#%% Training the actual network
inarg = (neural_net, train_data)
train_nn(*inarg)

#%% Model weight  save
neural_net.save_weights('../../../Dataset/Merl_Tim/NNsave/SavedWM/weights/my_checkpoint')

#%% Load weight load
neural_net.load_weights(
    '../../../Dataset/Merl_Tim/NNsave/SavedWM/weights/my_checkpoint')

#%% Random testing

i = 9

#trX1 = np.reshape(data[i:i+40,:,:,0], [40,100,100])
#trX1 = np.moveaxis(trX1, 0,-1)
#gt = pulR[i*2:i*2+80]

trX1 = trX[i]

gt = trY[i]

# trX1 = teX[i]

# gt = teY[i]


trX1 = np.reshape(trX1, [-1, 100,100,40])
plt.plot(gt)

trX1 = (trX1 - trX1.min())/(trX1.max() - trX1.min())

predd = neural_net(trX1) 
plt.plot(predd[0])

#%%  Extras

for i in range(40):
    print(i)
    plt.imshow(trainX[100,:,:,i])
    plt.show()
    
    

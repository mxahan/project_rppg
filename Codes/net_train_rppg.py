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
iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
dataPath = os.path.join(iD_ir, '*.pgm')
files = glob.glob(dataPath)
# end load pathdir

# load images  from 1 subject
data = []
im_size = (50,50)


for f1 in files:
    img =  cv2.resize(cv2.imread(f1)[:,:,1], im_size)
    img = img[:,:,np.newaxis]
    data.append(img)
    
data =  np.array(data)
    
#%%
# load Mat file
x = loadmat('../../../Dataset/Merl_Tim/Subject1_still/PulseOX/pulseOx.mat')

pulseoxR = np.squeeze(x['pulseOxRecord'])

pulR = []
for i in range(pulseoxR.shape[0]):
    pulR.append(pulseoxR[i][0][0])
    
pulR = np.array(pulR)
    
#%% Prepare dataset for training
random.seed(1)

rv = [randint(0,5300) for _ in range(10000)]
randint(0,5350)
rv =  np.array(rv)
pulR = np.reshape(pulR, [10703,1])
#      #%%

trainX = []
#trainY = np.zeros([200,80])
trainY = []


for j,i in enumerate(rv):
    img = np.reshape(data[i:i+40,:,:,0], [40,50,50])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    ppg = pulR[2*i: 2*i+80,0]
    trainY.append(ppg)


trainX = np.array(trainX, dtype = np.float32)
trainY = np.array(trainY, dtype = np.float32)

#%% Some parameter definition

num_classes = 80 # total classes (0-9 digits).
num_features = 50*50*40 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.0001
training_steps = 40000
batch_size = 8
display_step = 50


# Network parameters.


#%%    Normalize and split

trainX = (trainX-trainX.min())/(trainX.max()-trainX.min())
trainY = (trainY-trainY.min())/(trainY.max()-trainY.min())

trX, teX, trY, teY = train_test_split(trainX , trainY, test_size = .1, random_state = 42)

train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
train_data = train_data.repeat().shuffle(1).batch(batch_size).prefetch(1)



#%% Loss function  

def RootMeanSquareLoss(x,y):
    loss = tf.keras.losses.MSE(y_true = y, y_pred =x)
    #return tf.reduce_mean(loss) 
    return loss

#%%  Optimizer Definition
optimizer = tf.optimizers.SGD(learning_rate)

def run_optimization(neural_net, x,y):    
    with tf.GradientTape() as g:
        pred =  neural_net(x, is_training = True)
        loss =  RootMeanSquareLoss(pred, y)
        
    trainable_variables =  neural_net.trainable_variables
    
    gradients =  g.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
def train_nn(neural_net, train_data):
        
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):     
        run_optimization(neural_net, batch_x, batch_y)

        if step % display_step == 0:
            pred = neural_net(batch_x, is_training=True)
            loss = RootMeanSquareLoss(pred, batch_y)
            print("step: %i, loss: %f" % (step, tf.reduce_mean(loss)))
      

#%% Bringing Network
from net_work_def import ConvNet

#%% Training the actual network

neural_net = ConvNet(num_classes)
inarg = (neural_net, train_data)
train_nn(*inarg)

#%% Random testing

i = 900

trX1 = np.reshape(data[i:i+40,:,:,0], [40,50,50])
trX1 = np.moveaxis(trX1, 0,-1)

gt = pulR[i*2:i*2+80] 
plt.plot(gt/gt.max())

trX1 = (trX1 - trX1.min())/(trX1.max() - trX1.min())

predd = neural_net(trX1) 
plt.plot(predd[0])

#%%
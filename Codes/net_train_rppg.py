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
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/Merl_Tim'

subjects = ['/Subject1_still', '/Subject2_still', '/Subject3_still']

im_mode = ['/IR', '/RGB_raw', '/RGB_demosaiced']



iD_ir = path_dir + subjects[2]+im_mode[1]

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
x = loadmat(path_dir + subjects[2]+'/PulseOX/pulseOx.mat')

pulseoxR = np.squeeze(x['pulseOxRecord'])

pulR = []
for i in range(pulseoxR.shape[0]):
    pulR.append(pulseoxR[i][0][0])
    
pulR = np.array(pulR)
    
#%% Prepare dataset for training
# For subject one go till 5300
# For suject 2 go till 6230
# For subject 3 go till 7100
random.seed(1)

rv = [randint(0,7100) for _ in range(5000)]
rv =  np.array(rv)
pulR = np.reshape(pulR, [pulR.shape[0],1])
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
training_steps = 5000
batch_size = 8
display_step = 100


# Network parameters.


#%%    Normalize and split

trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
trainY = trainY/(trainY.max(axis = 1)[:, np.newaxis]+ 10**-5)

trainX = (trainX-trainX.min())/(trainX.max()-trainX.min())
#trainY = (trainY-trainY.min())/(trainY.max()-trainY.min()) # bad idea as global minima and outlines

trX, teX, trY, teY = train_test_split(trainX , trainY, test_size = .1, random_state = 42)

#%% tensorflow dataload

train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
train_data = train_data.repeat().shuffle(buffer_size=500, seed= 8).batch(batch_size).prefetch(1)


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
from net_work_def import ConvNet, ConvNet1
#%% load network
neural_net = ConvNet(num_classes)
# Basenet = ConvNet1(num_classes)
#%% Training the actual network
inarg = (neural_net, train_data)
with tf.device('gpu:0/'):
    train_nn(*inarg)

#%% Model weight  save
#neural_net.save_weights('../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/my_checkpoint')

#%% Load weight load
# neural_net.load_weights(
    # '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/my_checkpoint')

#%% Random testing

i = 93

# trX1 = np.reshape(data[i:i+40,:,:,:], [40,100,100])/255
# trX1 = np.moveaxis(trX1, 0,-1) # very important line in axis changeing 
# gt = pulR[i*2:i*2+80]
# gt = (gt-gt.min())/(gt.max()-gt.min())

trX1 = trX[i]

gt = trY[i]

# trX1 = teX[i]

# gt = teY[i]


trX1 = np.reshape(trX1, [-1, 100,100,40])
plt.plot(gt)

trX1 = (trX1 - trX1.min())/(trX1.max() - trX1.min())

predd = neural_net(trX1) 
plt.plot(predd[0])

#%% Seeing inside the network
in1 = neural_net.layers[0](trX1).numpy() # plt.plot(in1[0,:,:,1])
in2 = neural_net.layers[1](in1).numpy() # plt.plot(in2[0,:,:,1])  
in3 = neural_net.layers[2](in2).numpy()

# ##we can also select the model inside the inside layer

#neural_net.layers[0].layers[0].layers[1]

# Think of tensorflow 2 very flexible as long as we define the model very clearly

# Think of each of the layer and apply independently even when there is shared network inside...
# in2=neural_net.layers[0](trX1[:,:,:,10:20]).numpy()
# well we can go on

# ways to get the weights and biases. the layers should be in our mind in the firsthand 
#neural_net.layers[0].layers[0].layers[0].weights

#%%  Extras

for i in range(40):
    print(i)
    plt.imshow(trainX[100,:,:,i])
    plt.show()
    
    
#%% Get_weights and Set_weights
neural_net.layers[7].set_weights(Basenet.layers[7].get_weights())

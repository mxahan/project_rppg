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

path_dir = '../../../Dataset/datavideo/'

subjects = ['/sub10_me']

path_dir = path_dir + subjects[0]


dataPath = os.path.join(path_dir, '*.avi')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# load images  from 1 subject
#%% Load Video and load Mat file

data = []
im_size = (100,100)

cap = cv2.VideoCapture(files[0])

import pdb

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    gray  = gray[:,:,1]
    
    gray = cv2.resize(gray, im_size)
    
    # pdb.set_trace()
   
    data.append(gray)
    
    cv2.imshow('frame', gray)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


fps = cap.get(cv2.CAP_PROP_FPS)
    
cap.release()
cv2.destroyAllWindows()
    
data =  np.array(data)
    
#%% ppg loading
import pandas as pd

# x = np.array(pd.read_csv(path_dir + "/gtdump.xmp"))
# pulR = x[:,3]
# time_axis  = x[:,0] 

###### or the followings 

x = np.loadtxt(path_dir+"/ground_truth.txt")
pulR = x[0,:]
time_axis  = x[2,:] 

# ppg Hz 62.5!

# pulR = []
# for i in range(pulseoxR.shape[0]):
#     pulR.append(pulseoxR[i][0][0])  # check the inside shape sub 5 cause error
    
# pulR = np.array(pulR)
    
#%% Prepare dataset for training
# For subject 1,4 go till 5300
# For suject 2 go till 6230
# For subject 3 go till 7100

random.seed(1)
rv = np.arange(0,1800, 2)
# np.random.shuffle(rv)


# rv = [randint(0, 5300) for _ in range(5000)] ## random removal 
rv =  np.array(rv)
pulR = np.reshape(pulR, [pulR.shape[0],1]) # take 45 frames together
#      #%%

if 'trainX' in locals():
    print("already exists")
else:
    trainX = []
    trainY = []


data = data[:,:,:,np.newaxis]

frame_cons = 40 # how many frame to consider at a time

for j,i in enumerate(rv):
    
    img = np.reshape(data[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    # i = np.round(i/(fps*(time_axis[5]-time_axis[4]))) +1
    ppg = pulR[i: i+40,0]
    trainY.append(ppg)



#%% Some parameter definition

num_classes = 40
num_features = 100*100*40 

# Training parameters. Sunday, May 24, 2020 
learning_rate = 0.0002 # start with 0.001
training_steps = 50000
batch_size = 16
display_step = 100


# Network parameters.

# Multitask for each subjects
# varies the last layer (different last layers) 
# keep the backbone network same - Great idea
# create target from video itself. 

#%%    Normalize and split
trainX = np.array(trainX, dtype = np.float32)
trainY = np.array(trainY, dtype = np.float32)


trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
trainY = (trainY/(trainY.max(axis = 1)[:, np.newaxis]+ 10**-5))*2-1

trainX = (trainX-trainX.min())

trainX = trainX/ trainX.max()
#trainY = (trainY-trainY.min())/(trainY.max()-trainY.min())
 # bad idea as global minima and outlines

trX, teX, trY, teY = train_test_split(trainX , trainY, 
                                      test_size = .1, random_state = 42)


# for MTL
if 1==0:
    zercon = np.zeros(trY.shape)
    trY = np.concatenate([trY, zercon], axis = -1)
#%% tensorflow dataload

train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
train_data = train_data.repeat().shuffle(buffer_size=100,
                                         seed= 8).batch(batch_size).prefetch(1)



#%% MTL for second dataset (run till trainX and follow from here again)

trainX = np.array(trainX, dtype = np.float32)
trainY = np.array(trainY, dtype = np.float32)


trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
trainY = (trainY/(trainY.max(axis = 1)[:, np.newaxis]+ 10**-5))*2-1

trainX = (trainX-trainX.min())

trainX = trainX/ trainX.max()
#trainY = (trainY-trainY.min())/(trainY.max()-trainY.min())
 # bad idea as global minima and outlines

trX1, teX1, trY1, teY1 = train_test_split(trainX , trainY, 
                                      test_size = .1, random_state = 42)



# train_data1 = tf.data.Dataset.from_tensor_slices((trX1, trY1))
# train_data1 = train_data.repeat().shuffle(2000).batch(batch_size).prefetch(1)



#%% Loss function  
# import pdb

def RootMeanSquareLoss(x,y):
    loss = tf.keras.losses.MSE(y_true = y, y_pred =x)  # initial one
    #return tf.reduce_mean(loss)  # some other shape similarity
    # pdb.set_trace()   
    loss2 = tf.reduce_mean((tf.math.abs(tf.math.sign(y))-tf.math.sign(tf.math.multiply(x,y))),axis = -1)
    # print(loss2.shape)
    
    # print(tf.reduce_mean(loss), tf.reduce_mean(loss2))
    return loss + 0.25*loss2


#%%  Optimizer Definition
optimizer = tf.optimizers.SGD(learning_rate)
optimizer1 = tf.optimizers.SGD(learning_rate)

# def run_optimization(neural_net, x,y):    
#     with tf.GradientTape() as g:
#         pred =  neural_net(x, training = True)
#         loss =  RootMeanSquareLoss(y, pred)  # change for mtl
        
    
    
#     convtrain_variables =  neural_net.layers[0].trainable_variables
#     fcntrain_variables =  neural_net.layers[1].trainable_variables
    
#     # trainable_variables =  neural_net.trainable_variables[:-6] 
#     # also there are other ways to update the gradient it would give the same results
#     # trainable_var is a list, select your intended layers: use append
    
#     gradients =  g.gradient(loss, convtrain_variables+fcntrain_variables) 
#     # gradient and trainable variables are list
    
#     grads1 =  gradients[:len(convtrain_variables)]
#     grads2 = gradients[len(convtrain_variables):]
    
#     optimizer.apply_gradients(zip(grads1, convtrain_variables))
#     optimizer1.apply_gradients(zip(grads2, fcntrain_variables))
    
    # Or the following section or the above section 
    
def run_optimization(neural_net, x,y):    # for the second network varies in head
    with tf.GradientTape() as g:
        pred =  neural_net(x, training = True) 
        loss =  RootMeanSquareLoss(y, pred)  # change for mtl
        
    
    trainable_variables =  neural_net.trainable_variables
    # trainable_variables =  neural_net.trainable_variables[:-6] 
    # also there are other ways to update the gradient it would give the same results
    # trainable_var is a list, select your intended layers: use append
    
    gradients =  g.gradient(loss, trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, trainable_variables))


 
if 'train_loss' in locals():
    print("already exists")
else:
    train_loss =[]
    val_loss = []


def train_nn(neural_net1, neural_net2, train_data):
        
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):     
        run_optimization(neural_net1, batch_x, batch_y)
        
        # i = randint(0,trX1.shape[0]-20)
        # batch_x1 = tf.convert_to_tensor(trX1[i:i+batch_size])
        # batch_y1 = tf.convert_to_tensor(trY1[i:i+batch_size])
        # run_optimization(neural_net2, batch_x1, batch_y1)
        
        if step % (display_step*2) == 0:
            pred = neural_net1(batch_x, training=True)
            loss = RootMeanSquareLoss(batch_y, pred)
            train_loss.append(tf.reduce_mean(loss))
            Val_loss(neural_net1, teX[0:16], teY[0:16])
            print("step: %i, loss: %f val Loss: %f" % (step, tf.reduce_mean(loss), val_loss[-1]))
            
def Val_loss (neural_net, testX, testY):
    pred = neural_net(testX, training = False)
    loss = RootMeanSquareLoss(testY, pred)
    val_loss.append(tf.reduce_mean(loss))
    
    
#%% Bringing Network
from net_work_def import  MtlNetwork_head, MtlNetwork_body
# power of CNN
#%% load network
# neural_net = ConvNet(num_classes)
# Basenet = ConvNet1(num_classes) # No longer that important - too much parameters use others

mtl_body =  MtlNetwork_body()
head1 =  MtlNetwork_head(num_classes)
head2 = MtlNetwork_head(num_classes)

neural_net1 =  tf.keras.Sequential([mtl_body, head1])
neural_net2 =  tf.keras.Sequential([mtl_body, head2])


# Great result with multitasking model


#%% Training the actual network
# single net
# inarg = (neural_net, train_data)


# multi-task net
inarg = (neural_net1,neural_net2, train_data)


with tf.device('gpu:0/'):
    train_nn(*inarg)

#%% Model weight  save
# neural_net.save_weights('../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/sub4RGB_raw')
#my_checkpoint, sub3IR, sub1IR, sub4RGB_raw', sub3RGB_raw

#%% Load weight load
# neural_net.load_weights(
#       '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/sub1IR')

#%% Random testing

# modification in network 

# performance measurement. 

# peak penalize (except mse)

i = 811

fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for j in range( 1, columns*rows +1 ):
    
    i =randint( 5040, 5100)
    i=  1940 + j + j
    print(i)
    tX = np.reshape(data[i:i+40,:,:,:], [40,100,100])
    tX = np.moveaxis(tX, 0,-1) # very important line in axis changeing 
    # i = np.int(i/(fps*0.016)) +1
    gt = pulR[i: i+40,0]
    gt = (gt-gt.min())/(gt.max()-gt.min())
    
    
    # i  = 5+j +j
    # tX = teX[i]    
    # gt = 0.5*(teY[i]  +1)  
    # tX = teX[i]    
    # gt = 0.5*(teY[i]+1)

    
    fig.add_subplot(rows, columns, j)
    tX1 = np.reshape(tX, [-1, 100,100,40])
    plt.plot(gt*2-1)
    
    tX1 = (tX1 - tX1.min())/(tX1.max() - tX1.min())
    

    # predd = neural_net(trX1) 
    predd = neural_net1(tX1) 
    plt.plot(predd[0])
    plt.legend(["Ground Truth", "Predicted"])
    plt.xlabel('time')
    plt.ylabel('magnitude')
    
plt.show()



#%% Seeing inside the network
in1 = neural_net1.layers[0].layers[0](tX1).numpy() # plt.plot(in1[0,:,:,1])
in2 = neural_net1.layers[0].layers[1](in1).numpy() # plt.plot(in2[0,:,:,1])  
in3 = neural_net1.layers[0].layers[2](in2).numpy()

in4 = neural_net1.layers[0].layers[3](in3).numpy()
in5 = neural_net1.layers[0].layers[4](in4).numpy()
in6 = neural_net1.layers[0].layers[5](in5).numpy()
in7 = neural_net1.layers[0].layers[6](in6).numpy()
in8 = neural_net1.layers[0].layers[7](in7).numpy()
in9 = neural_net1.layers[0].layers[8](in8).numpy()

# in3 = neural_net.layers[2](in2).numpy()

# ##we can also select the model inside the inside layer

#neural_net.layers[0].layers[0].layers[1]

# Think of tensorflow 2 very flexible as long as we define the model very clearly

# Think of each of the layer and apply independently even when there is shared network inside...
# in2=neural_net.layers[0](trX1[:,:,:,10:20]).numpy()
# well we can go on

# ways to get the weights and biases. the layers should be in our mind in the firsthand 
#neural_net.layers[0].layers[0].layers[0].weights
# Keep track what you did in call and what you have in model layer definition

#%%  Extras

for i in range(40):
    print(i)
    plt.imshow(trainX[8000,:,:,i])
    plt.show()# create target from video itself. 

#%%    Normalize and split
trainX = np.array(trainX, dtype = np.float32)
trainY = np.array(trainY, dtype = np.float32)


trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
# trainY = trainY/(trainY.max(axis = 1)[:,np.newaxis]
    
    
#%% Get_weights and Set_weights

# neural_net.layers[7].set_weights(Basenet.layers[7].get_weights())
#weightss = np.array(neural_net.layers[0].layers[0].layers[0].weights)

#%% Plotting learning curves

tr_l = np.array(train_loss)

val_l = np.array(val_loss)

plt.plot(tr_l, 'r', val_l, 'g')

plt.xlabel("training step")

plt.ylabel("Errors in MSE")

plt.title("Learning curves for IR (person 3)")

lst = ["Training", 'Validation']

plt.legend(lst)


#%% Better visualization

fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for i in range(1, columns*rows +1):
    img = in8[0, :,:, 47+i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

#%% PPG visulization

plt.plot(pulR[500:4000])
plt.xlabel('time')
plt.ylabel('PPG magnitude')
plt.title("PPG magnitude changes")
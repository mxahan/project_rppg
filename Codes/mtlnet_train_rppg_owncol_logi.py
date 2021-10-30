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

import pandas as pd


#%%  Data Load Parts




# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/logi/zahid/'

ppgtotal =  pd.read_csv(path_dir +'zahid/BVP.csv')
EventMark = pd.read_csv(path_dir+'zahid/tags.csv')

dataPath = os.path.join(path_dir, '*.mov')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. 

#%% Load Video and load Mat file

# find start position by pressing the key position in empatica
# test1.MOV led appear at the 307th frame.

# perfect alignment! checked by (time_eM_last-time_eM_first)*30+start_press_sample  should
# give the end press in video



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
    gray =  gray[160:910, 725:1190]
   
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



#%% PPG signal selection and alignment. 
# The starting points are the crucial, 
# this section needs select both the sratrting of video and the ppg point

# check fps
# check starting time in BVP.csv
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[-2] -  1609184521

# check from BVP.csv column name. 
# Check video starting point from watching the frame with the light event marker 

end_point =  evmarknp[-1] - evmarknp[-2]

ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]

data_align = data[49 : 49 +np.int(end_point*30)+5] 

#%% Prepare dataset for training

# For subject 1,4 go till 5300
# For suject 2 go till 6230
# For subject 3 go till 7100

# 40 frames considered to to equivalent to 85 samples in PPg

random.seed(1)
rv = np.arange(0,6000, 1)+601
# np.random.shuffle(rv)


# rv = [randint(0, 5300) for _ in range(5000)] ## random removal 
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1]) # take 45 frames together
#      #%%

if 'trainX' in locals():
    print("already exists")
else:
    trainX = []
    trainY = []


data_align = data_align[:,:,:,np.newaxis]

frame_cons = 40 # how many frame to consider at a time

for j,i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    
    p_point = np.int(np.round(i*64/30))
    
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)



#%% Some parameter definition

num_classes = 85
num_features = 100*100*40 

# Training parameters. Sunday, May 24, 2020 
learning_rate = 0.0005 # start with 0.001
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

# we can run on epoch by removing repeat() and stating only batch and epoch number. 
# https://www.tensorflow.org/guide/data#batching_dataset_elements

#%% MTL for second dataset (run till trainX and follow from here again)

# trainX = np.array(trainX, dtype = np.float32)
# trainY = np.array(trainY, dtype = np.float32)


# trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
# trainY = (trainY/(trainY.max(axis = 1)[:, np.newaxis]+ 10**-5))*2-1

# trainX = (trainX-trainX.min())

# trainX = trainX/ trainX.max()
# #trainY = (trainY-trainY.min())/(trainY.max()-trainY.min())
# # bad idea as global minima and outlines

# trX1, teX1, trY1, teY1 = train_test_split(trainX , trainY, 
#                                       test_size = .1, random_state = 42)



# train_data1 = tf.data.Dataset.from_tensor_slices((trX1, trY1))
# train_data1 = train_data.repeat().shuffle(2000).batch(batch_size).prefetch(1)



#%% Loss function  


def RootMeanSquareLoss(x,y):
    
    # pdb.set_trace()  
    loss = tf.keras.losses.MSE(y_true = y, y_pred =x)  # initial one
    #return tf.reduce_mean(loss)  # some other shape similarity
     
    loss2 = tf.reduce_mean((tf.math.abs(tf.math.sign(y))-tf.math.sign(tf.math.multiply(x,y))),axis = -1)
    # print(loss2.shape)
    
    # print(tf.reduce_mean(loss), tf.reduce_mean(loss2))
    return loss + 0.5*loss2


#%%  Optimizer Definition
optimizer  = tf.optimizers.SGD(learning_rate)
optimizer1 = tf.optimizers.SGD(learning_rate/2)

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
    
    # # # # Or the following section 
    
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
        # pdb.set_trace()
        run_optimization(neural_net1, batch_x, batch_y)
        
        
        
        # i = randint(0,trX1.shape[0]-20)
        # batch_x1 = tf.convert_to_tensor(trX1[i:i+batch_size])
        # batch_y1 = tf.convert_to_tensor(trY1[i:i+batch_size])
        # run_optimization(neural_net2, batch_x1, batch_y1)
        
        
        if step % (display_step*2) == 0:
            pred = neural_net1(batch_x, training=True)
            # pdb.set_trace()
            loss = RootMeanSquareLoss(batch_y, pred)
            train_loss.append(tf.reduce_mean(loss))
            tp = np.random.randint(450)
            Val_loss(neural_net1, teX[tp+0:tp+16], teY[tp+0:tp+16])
            print("step: %i, loss: %f val Loss: %f" % (step, tf.reduce_mean(loss), val_loss[-1]))
            
def Val_loss (neural_net, testX, testY):
    pred = neural_net(testX, training = False)
    loss = RootMeanSquareLoss(testY, pred)
    val_loss.append(tf.reduce_mean(loss))
    
    
#%% Bringing Network
from net_work_def import  MtlNetwork_head, MtlNetwork_body
# power of CNN
#%% load network usual
# neural_net = ConvNet(num_classes)
# Basenet = ConvNet1(num_classes) # No longer that important - too much parameters use others

mtl_body =  MtlNetwork_body()
head1 =  MtlNetwork_head(num_classes)
head2 = MtlNetwork_head(num_classes)

neural_net1 =  tf.keras.Sequential([mtl_body, head1])
neural_net2 =  tf.keras.Sequential([mtl_body, head2])


# Great result with multitasking model


#%% Pruning Network 
from net_work_def import CNN_part

model__1 = CNN_part()

model__o1 = CNN_part()

model__2 = tf.keras.Sequential([tf.keras.layers.Dense(512, activation='relu', input_shape=(576,)),
                                tf.keras.layers.Dense(128, activation='relu', input_shape=(512,)),
                                tf.keras.layers.Dense(85, activation = 'tanh', input_shape=(128,))])


model__o2 = tf.keras.Sequential([tf.keras.layers.Dense(512, activation='relu', input_shape=(576,)),
                                tf.keras.layers.Dense(128, activation='relu', input_shape=(512,)),
                                tf.keras.layers.Dense(85, activation = 'tanh', input_shape=(128,))])


# model_f =  tf.keras.Sequential([model__1, model__2])

neural_net1 =  tf.keras.Sequential([model__o1, model__o2])
#%% Network to prune

neural_net2 =  tf.keras.Sequential([model__1, model__2])

#%% Training the actual network
# single net
# inarg = (neural_net, train_data)
# multi-task net

inarg = (neural_net1, neural_net1, train_data)

with tf.device('gpu:0'):
    train_nn(*inarg)

#%% Model weight  save

# model.set_inputs(tX1) is really important


input("saving Check the name again to save as it may overload previous .....")

model.save_weights('../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/zahid_keras')



###my_checkpoint, test1, emon_withglass, emon_withoutgss, sreeni2, emon_lab, avijoy, masud, zahid_logi, emon_logi, zahid_logi_prune


#%% Load weight load

input("loading Check before loading as it may overload previous .....")

model.load_weights(
        '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/zahid_keras')

#%% full model pruning 

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
  }

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

pruned_model = prune_low_magnitude(model, **pruning_params)

# Use smaller learning rate for fine-tuning
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

pruned_model.compile(
  loss=RootMeanSquareLoss1,
  optimizer=opt,
  metrics=['mse'])

pruned_model.summary()


#%% FTune

pruned_model.fit(
  train_data,
  epochs=3,
  # validation_split=0.1,
  callbacks=callbacks)

#%% helper 

def print_model_weights_sparsity(model):

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        for weight in weights:
            if "kernel" not in weight.name or "centroid" in weight.name:
                continue
            weight_size = weight.numpy().size
            zero_num = np.count_nonzero(weight == 0)
            print(
                f"{weight.name}: {zero_num/weight_size:.2%} sparsity ",
                f"({zero_num}/{weight_size})",
            )
            
stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

print_model_weights_sparsity(stripped_pruned_model)

stripped_pruned_model_copy = tf.keras.models.clone_model(stripped_pruned_model)
stripped_pruned_model_copy.set_weights(stripped_pruned_model.get_weights())

#%% pruning attempt OLDER
import tensorflow_model_optimization as tfmot

tfmot.sparsity.keras.prune_low_magnitude

epochs = 20

end_step =  180*epochs

# pruning_params = {
#       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.4,
#                                                                final_sparsity=0.80,
#                                                                begin_step=0,
#                                                                end_step=end_step)
# }

pruning_params = {
   'pruning_schedule' : tfmot.sparsity.keras.ConstantSparsity(0.9, begin_step=0, frequency=100)
    }

# model_for_p = tf.keras.Sequential([tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(512, activation='relu', input_shape=(576,))),
#                                 tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(128, activation='relu', input_shape=(512,))),
#                                 tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(85, activation = 'tanh', input_shape=(128,)))])

model_for_p = tfmot.sparsity.keras.prune_low_magnitude(model__2, **pruning_params)


model_f_prune = tf.keras.Sequential([model__1, model_for_p])


model_for_pruning.compile(optimizer='adam',
              loss=RootMeanSquareLoss,
              metrics=['mse'])

model_for_pruning.summary()


#%% Pruning fine tune

model_f_prune.compile(optimizer=optimizer,
              loss=RootMeanSquareLoss,
              metrics=['mse'])

for i in range(6):
    model_f_prune.fit(trX[i*900:(i+1)*900], trY[i*900:(i+1)*900],
                  batch_size=5, epochs=epochs,
              callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])


def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)
# wow so far works

#%% Pruning Continue
model_for_export =  tfmot.sparsity.keras.strip_pruning(model_for_p)

neural_net4 =  tf.keras.Sequential([model__1, model_for_export])

neural_net4.compile(optimizer='adam',
              loss=RootMeanSquareLoss,
              metrics=['mse'])

#%% prune official guide
import tempfile

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

#%% poly decay prune

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 10
validation_split = 0.1 # 10% of training set will be used for validation set. 

end_step = 338* epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.70,
                                                                begin_step=0,
                                                                end_step=end_step)
}

# pruning_params = {
#       'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.9, begin_step=0, frequency=100)
#   }


model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=RootMeanSquareLoss1,
              metrics=['mse'])

model_for_pruning.summary()

#%%

import tempfile

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_data, epochs=epochs,
                  callbacks=callbacks)

#%%
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

#%%

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

#%%

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

#%%
print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))

#%%

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')
with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))


#%% Random testing

# modification in network 

# performance measurement. 

# peak penalize (except mse)

# Don't forget to align HERE!! 

i = 5000

fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for j in range( 1, columns*rows +1 ):
    
    i=  1055+j*20
    print(i)
    tX = np.reshape(data_align[i:i+40,:,:,:], [40,100,100])
    tX = np.array(tX, dtype= np.float64)
    tX = np.moveaxis(tX, 0,-1) # very important line in axis changeing 
    
    p_point = np.int(np.round(i*64/30))
    
    gt = pulR[p_point: p_point+85, 0]

    gt = (gt-gt.min())/(gt.max()-gt.min())
    
    # i  = 5+j +j
    # tX = teX1[i]    
    # gt = 0.5*(teY1[i]+1)    
    # tX = teX[i]    
    # gt = 0.5*(teY[i]+1)
    
    fig.add_subplot(rows, columns, j)
    tX1 = np.reshape(tX, [-1, 100,100,40])
    plt.plot(gt*2-1)
    
    tX1 = (tX1 - tX1.min())/(tX1.max() - tX1.min())
    

    # predd = neural_net(trX1) 
    
    
    predd = model(tX1) 
    
    
    
    plt.plot(predd[0])

    plt.legend(["Ground Truth", "Predicted"])
    plt.xlabel('time sample \n (60 samples = 1 second)', fontsize =12)
    plt.ylabel('magnitude \n (Normalized voltage)', fontsize = 12)
    from matplotlib import rcParams
    rcParams['lines.linewidth'] = 2
    rcParams['lines.color'] = 'r'
   
# plt.savefig('sub4goodres.eps', format = 'eps', dpi= 600)
plt.show()





#%%  Extras

for i in range(40):
    print(i)
    plt.imshow(trainX[1000,:,:,i])
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

fig = plt.figure(figsize=(19.20,10.80))

# For suject 2 go till 6230
# For subject 3 go till 7100
plt.plot(tr_l, 'r', val_l, 'g')

plt.xlabel("training step")

plt.ylabel("Errors in MSE")

plt.title("Sample Learning Curve")

lst = ["Training", 'Validation']


plt.legend(lst)


plt.savefig('learning_curve.eps', format = 'eps', dpi= 1000)

#%% Better visualization

fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for i in range(1, columns*rows +1):
    img = in4[0, :,:, 20+i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)

# plt.savefig('inter_filt', dpi= 1000)
# plt.show()



#%% PPG and network visulization
neural_net1.layers[0].summary()
plt.plot(pulR[500:4000])
plt.xlabel('time')
plt.ylabel('PPG magnitude')
plt.title("PPG magnitude changes")

#%% Signal Reconstruction

divVec = np.ones([85])
divVec1 = np.zeros([85]) 

gtV = np.zeros([85])

recPPG = np.zeros([85])

for j in range(5):
    
    olap = 40
    i = 5080 -20+j*olap
    print(i)
    tX = np.reshape(data_align[i:i+40:1,:,:,:], [40,100,100])
    tX = np.array(tX, dtype= np.float64)
    tX = np.moveaxis(tX, 0,-1) # very important line in axis changeing 
     
    p_point = np.int(np.round(i*64/30))
    
    gt = pulR[p_point: p_point+85:1, 0]
    gt = (gt-gt.min())/(gt.max()-gt.min())
    
    # i  = 5+j +j
    # tX = teX1[i]    
    # gt = 0.5*(teY1[i]+1)    
    # tX = teX[i]    
    # gt = 0.5*(teY[i]+1)    

    tX1 = np.reshape(tX, [-1, 100,100,40])

    tX1 = (tX1 - tX1.min())/(tX1.max() - tX1.min())
    
    olap =  np.int(olap*64/30)
    # predd = neural_net(trX1) 
    predd = model(tX1) 
    recPPG[-85:] = recPPG[-85:] + predd
    recPPG = np.concatenate((recPPG, np.zeros([olap])))
    
    
    gtV[-85:] = gtV[-85:] + np.squeeze(gt*2-1)
    gtV = np.concatenate((gtV, np.zeros([olap])))
    
    
    divVec1[-85:] = divVec1[-85:]+divVec
    divVec1 = np.concatenate((divVec1, np.zeros([olap])))    
    
    
    


fig = plt.figure(figsize=(19.20,10.80))
plt.plot(gtV[:-85], 'C2')
plt.plot(recPPG[:-85], 'C3')
plt.legend(["Ground Truth", "Predicted"], fontsize = 42, loc = "upper right", ncol = 2)
plt.xlabel('time sample (60 samples = 1 second)', fontsize =40, fontweight = 'bold')
plt.ylabel('PPG magnitude \n (Normalized voltage)', fontsize = 40, fontweight= 'bold')

plt.title("PPG plot for the collected data", fontsize = 40, fontweight = 'bold')

plt.margins(x =0, y =0.17)
# from matplotlib import rcParams
# rcParams['lines.linewidth'] = 4
# rcParams['lines.color'] = 'r'
plt.xticks(fontsize = 35, fontweight = 'bold')
plt.yticks(fontsize = 35, fontweight =  'bold')

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['font.weight'] = 200
plt.style.use('seaborn-whitegrid')

mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.linewidth'] = 5
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['legend.fontsize'] = 14

# plt.savefig('cd_sample_res.eps', format = 'eps', dpi= 500)

plt.show() 

#%% Tensorflow lite conversion
# path selection and save original model
# need to _set_inputs(tX1) before final compression 
new_path =  os.path.join("../../../Dataset/Merl_Tim/NNSave/SavedWM")

tX1 = np.array(tX1, dtype=np.float32) # holy crap I need this line

model_for_pruning._set_inputs(tX1) # run this line once

#%% save model

# tf.saved_model.save(neural_net1, new_path)# this guy not works???
# neural_net1.save(new_path)
# tf.keras.models.save_model(neural_net1, new_path)
# save as assets, variable, .pb file extension 

# tf lite converter
# conMod = tf.lite.TFLiteConverter.from_saved_model(new_path)
# 
# alternatively we can get from the model itself without any saving!!!

conMod = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
# tflite_model = converter.convert()


# conMod.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                         tf.lite.OpsSet.SELECT_TF_OPS] # need this for some reason

tfLitMod =  conMod.convert()

# neural_net2.layers[0].summary()

#%% lite model save both all and FP16

# save the lit model
# write the model to the ***.tflite format!
import pathlib

tflite_models_dir = pathlib.Path("../../../Dataset/Merl_Tim/NNSave/SavedWM")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"emon_lite_p40.tflite"
tflite_model_file.write_bytes(tfLitMod)

# further optimization

conMod.optimizations = [tf.lite.Optimize.DEFAULT]
conMod.target_spec.supported_types = [tf.float16]

# save the model in **.tflite format

tflite_fp16_model = conMod.convert()
tflite_model_fp16_file = tflite_models_dir/"emon_lit_p_f1640.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)


#%% inference bring the tflite_model_file

# bring and allocate-tensors
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
# alternatively

# interpreter = tf.lite.Interpreter(model_content=tfLitMod)

interpreter.allocate_tensors()

# see inside
print(interpreter.get_input_details())
print(interpreter.get_output_details())
# port for the input and output

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# format input_data
# interpreter set tensor
# we can provide intermediate data too by selecing the index
input_data = np.array(tX1, dtype=np.float32)
interpreter.set_tensor(input_index, input_data)

# getting out the output, 
# allocate tensor, fill values before invoke
interpreter.invoke()
# have everything in # get_tensor_details()
# all intermediate results are there.
# get_tensor to get some tensor from the desired index
# mostly we want output_index
predictions = interpreter.get_tensor(output_index)

plt.plot(predictions.reshape([85]), 'C2')
plt.plot(recPPG[:-85], 'C3')

plt.show()
#%% Fp16 Inference
# same as previous but a more quantized model
# interpreter = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))

interpreter = tf.lite.Interpreter(model_content=tflite_fp16_model)
interpreter.allocate_tensors()

# see inside
print(interpreter.get_input_details())
print(interpreter.get_output_details())

input_index = interpreter.get_input_details()[0]["index"]

output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, tX1.astype(np.float32))

interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

plt.plot(predictions.reshape([85]))
plt.plot(gtV[:-85], 'C3')

#%% Space check

# ls -lh
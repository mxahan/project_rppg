#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:06:57 2020

@author: zahid
"""


#%% Load libraries
import tensorflow as tf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # CNN overload error

import matplotlib.pyplot as plt

import numpy as np

import cv2

import glob

#from scipy.io import loadmat

import random

from random import seed, randint

#from sklearn.model_selection import train_test_split

import pandas as pd
#%%  Data Load files from the directory

# Select the source file [either MERL or Collected Dataset or ]


# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/Personal_collection/MPSC_rppg/subject_005/trial_001/video/'



dataPath = os.path.join(path_dir, '*.MOV')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. 


#%% Load the Video and corresponding GT Mat file

# find start position by pressing the key position in empatica
# perfect alignment! checked by (time_eM_last-time_eM_first)*30+start_press_sample  should
# give the end press in video


def data_read(files, im_size = (200, 200)):
    data = []
    cap = cv2.VideoCapture(files[0])
    
    # import pdb
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret==False:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gray  = gray[:,:,1]
        gray =  gray[:900, 600:1500]
        gray = cv2.resize(gray, im_size)
        # pdb.set_trace()
        data.append(gray)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # fps = cap.get(cv2.CAP_PROP_FPS)
        
    cap.release()
    cv2.destroyAllWindows()
    data =  np.array(data)
    
    return data

data = data_read(files, im_size = (100, 100) )


#%% PPG signal selection and alignment. 

# The starting points are the crucial, 
# this section needs select both the sratrting of video and the ppg point
# check fps and starting time in BVP.csv
# Match the lines from the supplimentary text file for the data
def alignment_data(data, path_dir):
    ppgtotal =  pd.read_csv(path_dir +'../empatica_e4/BVP.csv')
    EventMark = pd.read_csv(path_dir+'../empatica_e4/tags.csv')
    evmarknp =  EventMark.to_numpy()
    ppgnp =  ppgtotal.to_numpy()
    start_gap =  evmarknp[0] -  1599849555
    
    end_point =  evmarknp[1] - evmarknp[0]
    
    ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
    
    data_align = data[176 : 176 +np.int(end_point*30)+5]  
    ppgnp_align = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1]) 

    return data_align, ppgnp_align

data_align, ppgnp_align = alignment_data(data, path_dir) 
del data

#%% pickle save
import pickle
input("pickle save ahead")
# Saving 
save_path  =  '../../../Dataset/Personal_collection/MPSC_rppg/Pickle_files_rppg/sub_002_003.pkl'
def pickle_save(data_align, ppgnp_align, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump([data_align, ppgnp_align], f)  #NickName: SAD
    

# Loading 
def load_pickle(save_path):
    with open(save_path, 'rb') as f:
        a,b = pickle.load(f)
        return a, b
    
#%% pickle load

#%% Prepare data loader
import numpy 

from random import choice
class data_loader():
    def __init__(self, video, ppg= None, im_size = (100, 100), frame_cons = 40, bs = 1):
        self.video = video
        self.max_len = video.shape[0] - 200
        self.ppg =  ppg
        self.im_size = im_size
        self.frame_cons = frame_cons
        self.bs = bs
    
    def get_sup_samp(self, pos  = None, frame_gap = 1):
        
        if pos==None:
            pos = randint(0, self.max_len)
        img = np.transpose(self.video[pos:pos+self.frame_cons*frame_gap:frame_gap,:,:],[1,2,0])
        p_point = np.int(np.round(pos*64/30))
        ppg_gt = self.ppg[p_point: p_point+85*frame_gap: frame_gap, 0]
        
        ppg_gt = ppg_gt-np.min(ppg_gt)
        
        ppg_gt = (ppg_gt/np.max(ppg_gt))*2 -1
        
        img =  self.img_resize(img)
        
        return img, ppg_gt, pos
    
    def get_sup_data(self, bs = None):
        vs, gt = [], []
        if bs ==None: bs =self.bs
        for i in range(bs):
            frame_gap= 1 if random.random()>0.90 else 2
            sas, gtss, _ =  self.get_sup_samp(frame_gap=frame_gap)
            vs.append(sas); gt.append(gtss)
        return tf.cast(tf.stack(vs), tf.float32)/255.0, tf.cast(tf.stack(gt), tf.float32)
    
    def rand_frame_shuf(self, img):
        img = img.numpy()
        temp = np.arange(40)
        np.random.shuffle(temp)
        return img[:,:, temp]
    
    def frame_repeat(self, img):
        img = img.numpy()
        temp = randint(0,39)
        img[:,:,0:40] = img[:,:, temp:temp+1]
        return img
    
    def img_resize(self, img):
        return tf.image.resize(img, self.im_size)
    
    def rand_crop_tf(self, img):
        [temp1, temp2] =[randint(120,180),randint(120,180)]
        img = tf.image.random_crop(img, size=(temp1, temp2, 40)).numpy()
        return self.img_resize(img)
    
    def fps_halfing(self, pos):
        img, ppg_gt, _ = self.get_sup_samp(pos= pos, frame_gap= 2)
        return img, ppg_gt
    
    def img_shifted(self, pos, sv):
        img, _, _ =  self.get_sup_samp(pos = pos+sv)   
        return img
    
    def get_CL_data(self, pve = 1, nve = 5):
        vs = []
        query, _, pos = self.get_sup_samp()
        vs.append(query)
        pos_op = [0, 1]
        for _ in range(pve):
            dum = choice(pos_op)
            if dum ==0:
                pq = self.img_shifted(pos = pos, sv = 26)
                vs.append(pq)
            elif dum ==1:
                pq = self.rand_crop_tf(query)
                vs.append(pq)
        
        neg_op = [0, 1, 2, 3]
        for _ in range(nve):
            dum = choice(neg_op)
            if dum==0:
                nq = self.img_shifted(pos = pos, sv = 13)
                vs.append(nq)
            elif dum==1:
                nq = self.frame_repeat(query)
                vs.append(nq)
            elif dum==2:
                nq = self.rand_frame_shuf(query)
                vs.append(nq)
            elif dum==3:
                nq, _ = self.fps_halfing(pos = pos)
                vs.append(nq)
                
        return  tf.cast(tf.stack(vs), tf.float32)/255.0
 

#%% MTL for second dataset (run till trainX and follow from here again)
# Make sure this is connected to person 2

# trainX = np.array(trainX, dtype = np.float32)
# trainY = np.array(trainY, dtype = np.float32)


# trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
# trainY = (trainY/(trainY.max(axis = 1)[:, np.newaxis]+ 10**-5))*2-1

# trainX = (trainX-trainX.min())

# trainX = trainX/ trainX.max()

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

def RootMeanSquareLoss1(y,x):
    
    # pdb.set_trace()  
    loss = tf.keras.losses.MSE(y_true = y, y_pred =x)  # initial one
    #return tf.reduce_mean(loss)  # some other shape similarity
     
    loss2 = tf.reduce_mean((tf.math.abs(tf.math.sign(y))-tf.math.sign(tf.math.multiply(x,y))),axis = -1)
    # print(loss2.shape)
    
    # print(tf.reduce_mean(loss), tf.reduce_mean(loss2))
    return loss + 0.5*loss2


#%% Bringing Network
from net_work_def import  MtlNetwork_head, MtlNetwork_body

def netw_fcnal(num_classes = 85):
    mtl_body =  MtlNetwork_body()
    head1 =  MtlNetwork_head(num_classes)
    head2 = MtlNetwork_head(num_classes)
    
    neural_net1 =  tf.keras.Sequential([mtl_body, head1])
    neural_net2 =  tf.keras.Sequential([mtl_body, head2])
    return neural_net1, neural_net2


# Great result with multitasking model


#%%  Optimizer Definition




# select the network portion to train [need in partial training]
def run_optimization(neural_net, x,y, learning_rate = 1e-3):   
    optimizer  = tf.optimizers.SGD(learning_rate*2)
    optimizer1 = tf.optimizers.SGD(learning_rate/2)
    with tf.GradientTape() as g:
        pred =  neural_net(x, training = True)
        loss =  RootMeanSquareLoss(y, pred)  # change for mtl
    
    convtrain_variables =  neural_net.layers[0].trainable_variables
    fcntrain_variables =  neural_net.layers[1].trainable_variables
    
    # trainable_variables =  neural_net.trainable_variables[:-6] 
    # also there are other ways to update the gradient it would give the same results
    # trainable_var is a list, select your intended layers: use append
    
    gradients =  g.gradient(loss, convtrain_variables+fcntrain_variables) 
    # gradients and trainable variables are list
    
    grads1 =  gradients[:len(convtrain_variables)]
    grads2 = gradients[len(convtrain_variables):]
    
    optimizer.apply_gradients(zip(grads1, convtrain_variables))
    optimizer1.apply_gradients(zip(grads2, fcntrain_variables))
    
    # # # # Or the following section 
    
# def run_optimization(neural_net, x,y):    # for the second network varies in head
#     with tf.GradientTape() as g:
#         pred =  neural_net(x, training = True) 
#         loss =  RootMeanSquareLoss(y, pred)  # change for mtl
#     trainable_variables =  neural_net.trainable_variables
#     # trainable_variables =  neural_net.trainable_variables[:-6] 
#     # also there are other ways to update the gradient it would give the same results
#     # trainable_var is a list, select your intended layers: use append  
#     gradients =  g.gradient(loss, trainable_variables)  
#     optimizer.apply_gradients(zip(gradients, trainable_variables))


 
if 'train_loss' in locals():
    print("already exists")
else:
    train_loss =[]
    val_loss = []


def train_nn(neural_net1, neural_net2, train_data):
 
        
    for step in range(5000): 
        # pdb.set_trace()
        
        # body + Head1 training
        
        
        batch_x, batch_y = train_data.get_sup_data()
        run_optimization(neural_net1, batch_x, batch_y)
        
        # run_optimization(neural_net2, batch_x1, batch_y1)
        
        
        if step % (100*2) == 0:
            pred = neural_net1(batch_x, training=True)
            # pdb.set_trace()
            loss = RootMeanSquareLoss(batch_y, pred)
            train_loss.append(tf.reduce_mean(loss))
            # Val_loss(neural_net1, teX[0:16], teY[0:16])
            print("step: %i, loss: %f v" % (step, tf.reduce_mean(loss)))
            
def Val_loss (neural_net, testX, testY):
    pred = neural_net(testX, training = False)
    loss = RootMeanSquareLoss(testY, pred)
    val_loss.append(tf.reduce_mean(loss))



with tf.device('gpu:0/'): 
    samp_load =  data_loader(video=data_align, ppg = ppgnp_align, bs = 8)
    neural_net1, neural_net2 =  netw_fcnal()
    train_nn(neural_net1, neural_net2, train_data = samp_load)
    del samp_load
    
    

#%% Model weight  save

input("Check the name again to save as it may overload previous .....")

# neural_net1.save_weights('../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/random_name change name')

# 

###my_checkpoint, test1, emon_withglass, emon_withoutglass, sreeni2

#%% Load weight load

input("Check before loading as it may overload previous .....")

# neural_net3.load_weights(
#         '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/rini1')



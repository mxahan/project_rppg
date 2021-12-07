#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:06:57 2020

@author: zahid
"""
#%% Load libraries


import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # CNN overload error


import numpy as np

import cv2

import glob

import pandas as pd
#%%  Data Load files from the directory

# Select the source file [either MERL or Collected Dataset or ]


# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/Personal_collection/MPSC_rppg/subject_001/trial_001/video/'

ppgtotal =  pd.read_csv(path_dir +'../empatica_e4/BVP.csv')
EventMark = pd.read_csv(path_dir+'../empatica_e4/tags.csv')

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
    #update the following ling
    gray =  gray[10:900, 720:1500]
    
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
# check fps and starting time in BVP.csv
# Match the lines from the supplimentary text file for the data

evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()


start_gap =  evmarknp[0] - 1593893213 
end_point =  evmarknp[1] - evmarknp[0] # default (..[1] -..[0])
data_align = data[307:307+np.int(end_point*30)+5]


ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
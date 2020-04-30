#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:06:08 2020

@author: zahid
"""

from tensorflow.keras import Model, layers
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def convlayer():
    convla = layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu)
    return convla


class ConvNet(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.bn1 = layers.BatchNormalization()
        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

    
        
        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.concat1 = layers.Concatenate()

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes, activation=tf.nn.relu)

    # Set forward pass.
    def call(self, x, is_training=None):
        x = tf.reshape(x, [-1, 50, 50, 40])
        xl = []
        for i in range(4):
            x1 = x[:,:,:,i*10:(i+1)*10]
            x2 = self.conv1(x1)
            xl.append(x2)
        
        

        
        # print(x1.shape)
        
        x =self.concat1(xl)
        
        x = self.bn1(x, training = is_training)
        
        # print(x.shape)
        
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out(x)

        return x

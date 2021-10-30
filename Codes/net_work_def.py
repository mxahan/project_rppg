#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:06:08 2020

@author: zahid
"""
# Important resource https://github.com/dragen1860/TensorFlow-2.x-Tutorials

from tensorflow.keras import Model, layers
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class ConvBNRelu(keras.Model):
    
    def __init__(self, ch, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            layers.Conv2D(ch, kernel_size =  kernel_size, strides=strides, padding=padding,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        
    def call(self, x, training=None):
        
        x = self.model(x, training=training)
        
        return x 
    
# inception module

class InceptMod(keras.Model):
    def __init__(self, ch, strides = 1):
        super(InceptMod, self).__init__()
        
        self.ch = ch
        self.strides = strides
        
        self.conv1 = ConvBNRelu(ch, kernel_size=1, strides=strides)
        
        self.conv2 = ConvBNRelu(ch, kernel_size=3, strides=strides)
        
        self.conv3_1 = ConvBNRelu(ch, kernel_size=5,  strides=strides)
        
        self.conv3_2 = ConvBNRelu(ch, strides=strides)
        
        
        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)
        
        
    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)
                
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
                
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)
        
        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)
        
        return x
    
#%% Initial Network works fine but 41M  parameters the problem with dense layers

class ConvNet1(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self, num_classes):
        super(ConvNet1, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = ConvBNRelu(32)
        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = ConvBNRelu(64)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = ConvBNRelu(32, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and aBasenet kernel size of 3.
        self.conv4 = ConvBNRelu(64, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1024, activation=tf.nn.relu)
        
        self.fc2 = layers.Dense(512, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).
        # self.concat1 = layers.Concatenate()

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes, tf.nn.relu)

    # Set forward pass.
    def call(self, x, training=False):
  #       x = tf.reshape(x, [-1, 100, 100, 40])
        # xl = []
        # for i in range(4):
        #     x1 = x[:,:,:,i*10:(i+1)*10]
        #     x2 = self.conv1(x1, training=training)
        #     xl.append(x2)
        

        # # print(x1.shape)
        
        # x =self.concat1(xl)
        
        x = self.conv1(x, training=training)
        # print(x.shape)
        
        x = self.conv2(x, training=training)
        x = self.maxpool1(x)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        return x

# Network Option  2 

class ConvNet(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = ConvBNRelu(32)
        
        # self.concat1 = layers.Concatenate()

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = ConvBNRelu(64)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = ConvBNRelu(64, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = ConvBNRelu(96, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        
        self.incept1 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        
        self.incept2 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool2 = layers.AveragePooling2D(2, strides= 2)
        
        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        
        self.fc2 = layers.Dense(512, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, training=False):
 #       x = tf.reshape(x, [-1, 100, 100, 40])
        
        # x =self.concat1(xl)
        
        x = self.conv1(x, training=training)
        # print(x.shape)
        x = self.conv2(x, training=training)
        x = self.maxpool1(x)
        
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        x = self.maxpool2(x)
        
        x = self.incept1(x, training = training)
        
        x = self.avgpool1(x)
   
        x = self.incept2(x, training = training)
        
        x = self.avgpool2(x)
        # print(x.shape)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = tf.nn.tanh(x)

        return x



# Network Option  2 

class MtlNetwork(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self, num_classes):
        super(MtlNetwork, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = ConvBNRelu(32)
        
        # self.concat1 = layers.Concatenate()

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = ConvBNRelu(64)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = ConvBNRelu(64, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = ConvBNRelu(96, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        
        self.incept1 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        
        self.incept2 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool2 = layers.AveragePooling2D(2, strides= 2)
        
        self.flatten = layers.Flatten()

        self.fc1_1 = layers.Dense(512, activation=tf.nn.relu)
        
        self.fc1_2 = layers.Dense(512, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).

        # Output layer, class prediction.
        self.out1 = layers.Dense(num_classes)
        
        
        
        self.fc2_1 = layers.Dense(512, activation=tf.nn.relu)
        
        self.fc2_2 = layers.Dense(512, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).

        # Output layer, class prediction.
        self.out2 = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, training=False):
 #       x = tf.reshape(x, [-1, 100, 100, 40])
        
        # x =self.concat1(xl)
        
        x = self.conv1(x, training=training)
        # print(x.shape)
        x = self.conv2(x, training=training)
        x = self.maxpool1(x)
        
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        x = self.maxpool2(x)
        
        x = self.incept1(x, training = training)
        
        x = self.avgpool1(x)
   
        x = self.incept2(x, training = training)
        
        x = self.avgpool2(x)
        # print(x.shape)
        
        x = self.flatten(x)
        
        
        x1 = self.fc1_1(x)
        x1 = self.fc1_2(x1)
        x1 = self.out1(x1)
        x1 = tf.nn.tanh(x1)
        
        
        x2 = self.fc2_1(x)
        x2 = self.fc2_2(x2)
        x2 = self.out2(x2)
        x2 = tf.nn.tanh(x2)
        
        x3 =  layers.concatenate([x1, x2])
        
        # print(x3.shape)

        return x1, x2


# MTL body head

#%% Actual Network starts from Here. 

# Body network
class MtlNetwork_body(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self):
        super(MtlNetwork_body, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = ConvBNRelu(32)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = ConvBNRelu(64)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = ConvBNRelu(64, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = ConvBNRelu(96, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        
        self.incept1 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        
        self.incept2 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool2 = layers.AveragePooling2D(2, strides= 2)
        
        self.flatten = layers.Flatten()
        
        self.fc1_1 = layers.Dense(512, activation=tf.nn.relu)
        
        self.fc1_2 = layers.Dense(128, activation=tf.nn.relu) # changed from 512


    # Set forward pass.
    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 100, 100, 40])
        
        # x =self.concat1(xl)
        
        x = self.conv1(x, training=training)
        # print(x.shape)
        x = self.conv2(x, training=training)
        x = self.maxpool1(x)
        
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        x = self.maxpool2(x)
        
        x = self.incept1(x, training = training)
        
        x = self.avgpool1(x)
   
        x = self.incept2(x, training = training)
        
        x = self.avgpool2(x)
        # later added        
        x = self.avgpool2(x)
        
        
        # later added        

        # print(x.shape)
        
        x = self.flatten(x)
        
        x = self.fc1_1(x)
        x = self.fc1_2(x)
        
        

        
        # print(x3.shape)

        return x



# Head network

class MtlNetwork_head(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self, num_classes):
        super(MtlNetwork_head, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.


        # Apply Dropout (if is_training is False, dropout is not applied).

        # Output layer, class prediction.
        self.out1 = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, training=False):
        

        x = self.out1(x)
        x = tf.nn.tanh(x)
        # print(x3.shape)
        return x
    

#%% 
# with umbc vpn
# https://gpvpn.umbc.edu/https/dl.acm.org/doi/pdf/10.1145/3356250.3360036

class VitaMon1(Model):
    def __init__(self, num_classes):
        super(VitaMon1, self).__init__()
        self.conv1 =  layers.Conv2D(32, kernel_size = 3, strides=1, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv2 =  layers.Conv2D(32, kernel_size = 3, strides=1, padding="same",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv3 =  layers.Conv2D(64, kernel_size = 3, strides=1, padding="same",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        self.conv4 =  layers.Conv2D(80, kernel_size = 3, strides=1, padding="same",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv5 =  layers.Conv2D(192, kernel_size = 3, strides=1, padding="same",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        
        self.incept1 = InceptMod(ch = 16, strides = 1)
        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        self.flatten = layers.Flatten()

        self.drop1 =  layers.Dropout(0.5)
        self.out1 = layers.Dense(num_classes, activation=tf.nn.relu)
        
        
    def call(self, x, training=False ):
        x = tf.reshape(x, [-1,100,100,40])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        
        # print(x.shape)
        x = self.maxpool1(x)
        
        x = self.incept1(x)
        
        x = self.avgpool1(x)
        x = self.avgpool1(x)
        x = self.avgpool1(x)
        x = self.avgpool1(x)
        
        x = self.drop1(x)
        
        x = self.flatten(x)
        
        x = self.out1(x)
                
        return x
    
    

class VitaMon2(Model):
    def __init__(self, num_classes):
        super(VitaMon1, self).__init__()
        self.conv1 =  layers.Conv2D(32, kernel_size = 3, strides=1, padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv2 =  layers.Conv2D(32, kernel_size = 3, strides=1, padding="same",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv3 =  layers.Conv2D(64, kernel_size = 3, strides=1, padding="same",
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        self.flatten = layers.Flatten()

        self.drop1 =  layers.Dropout(0.5)
        self.out1 = layers.Dense(num_classes)
        
        
    def call(self, x, training=False ):
        x = tf.reshape(x, [-1,100,100,11])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        

        x = self.avgpool1(x)
        x = self.avgpool1(x)
        x = self.avgpool1(x)
        x = self.avgpool1(x)
        
        x = self.drop1(x)
        
        x = self.flatten(x)
        
        x = self.out1(x)
                
        return x
    
    #%% Pruning modified network
    
class CNN_part(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self):
        super(CNN_part, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = ConvBNRelu(32)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = ConvBNRelu(64)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = ConvBNRelu(64, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = ConvBNRelu(96, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        
        self.incept1 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        
        self.incept2 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool2 = layers.AveragePooling2D(2, strides= 2)
        
        self.flatten = layers.Flatten()
            # Set forward pass.
            
    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 100, 100, 40])
        
        # x =self.concat1(xl)
        
        x = self.conv1(x, training=training)
        # print(x.shape)
        x = self.conv2(x, training=training)
        x = self.maxpool1(x)
        
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        x = self.maxpool2(x)
        
        x = self.incept1(x, training = training)
        
        x = self.avgpool1(x)
   
        x = self.incept2(x, training = training)
        
        x = self.avgpool2(x)
        # later added        
        x = self.avgpool2(x)     
        
        x = self.flatten(x)

        
        return x
    
    

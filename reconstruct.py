# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras import metrics
from keras.optimizers import Adam 

from data import *
from model import *
from matplotlib.image import imsave

rare_classes = [6,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,27]

input_shape = (256, 256, 3)

keras.backend.clear_session()

# Datasets
train_data_path = '../train/'
label_path = '../train.csv'
train_dataset_info = read_train_info(label_path, train_data_path)

recons_path = 'recons_image'
batch_size = 32

# create model
autoencoder_model = segnet_model(input_shape=input_shape, n_labels=28)

adam = Adam(lr=1e-4)

autoencoder_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['acc'])

autoencoder_model.load_weights('segnet_autoencoder.h5')

for i in range(len(train_dataset_info) // batch_size):
    print(str(i) + "th batches")
    mini_batch_info = train_dataset_info[i:i+batch_size]
    
    for img_info in mini_batch_info:
        if not any([x in rare_classes for x in [label for label in img_info['labels']]]):
            continue
        orig_img = data_generator.load_image(img_info['path'], input_shape)
        recons_img = autoencoder_model.predict(orig_img[np.newaxis, :, :, :] / 255.)
        imsave(recons_path+'/'+img_info['path'].split('/')[-1]+'.png', recons_img[0,:,:,:])

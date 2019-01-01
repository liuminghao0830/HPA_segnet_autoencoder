# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from data import *
from model import *

K.set_image_data_format('channels_last')


# Start training model
input_shape = (256, 256, 3)

keras.backend.clear_session()

# Datasets
train_data_path = '../../train/'
label_path = '../train.csv'
train_dataset_info = read_train_info(label_path, train_data_path)

# create callbacks list
epochs = 20; batch_size = 16
checkpoint = ModelCheckpoint('segnet_autoencoder.h5', monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)
#reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
#                                   verbose=1, mode='auto', epsilon=0.0001)
#early = EarlyStopping(monitor="val_loss", 
#                      mode="min", 
#                      patience=6)
#callbacks_list = [checkpoint, early, reduceLROnPlat]

# split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)

# create train and valid datagens
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, input_shape, augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 16, input_shape, augument=False)


# create model
train_model = segnet_model(input_shape=input_shape, n_labels=28)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.003)

train_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['acc'])

train_model.fit_generator(train_generator,
            steps_per_epoch=np.floor(float(len(train_indexes)) / float(batch_size)),
            validation_data=next(validation_generator),
            validation_steps=np.floor(float(len(valid_indexes)) / float(batch_size)),
            epochs=epochs, verbose=1, callbacks=[checkpoint])

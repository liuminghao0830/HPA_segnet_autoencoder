# -*- coding: utf-8 -*-

import numpy as np
import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

K.set_image_data_format('channels_last')


def segnet_model(input_shape, n_labels):
    kernel = 3
    pool_size = 2

    encoding_layers = [
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),

        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size, strides=2),
    ]

    decoding_layers = [
        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(128, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(64, kernel_size=kernel, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(input_shape[2], kernel_size=1, strides=1, padding='valid'),
        BatchNormalization(),
    ]


    segnet_basic = Sequential()

    segnet_basic.add(Layer(input_shape=input_shape))


    segnet_basic.encoding_layers = encoding_layers
    for l in segnet_basic.encoding_layers:
        segnet_basic.add(l)


    segnet_basic.decoding_layers = decoding_layers
    for l in segnet_basic.decoding_layers:
        segnet_basic.add(l)

    segnet_basic.summary()
    return segnet_basic


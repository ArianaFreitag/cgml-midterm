#!/bin/python3.6
'''
Implementation of Resnet-50 from "Deep Residual Learning for Image Recognition" (https://arxiv.org/pdf/1512.03385.pdf)
and "Identity Mappings in Deep Residual Networks" (https://arxiv.org/abs/1603.05027.pdf). More specicifally
our block will look like this https://raw.githubusercontent.com/raghakot/keras-resnet/master/images/residual_block.png

The architecture for 50-layer looks like 
this https://raw.githubusercontent.com/raghakot/keras-resnet/master/images/architecture.png (one in the middle)
Minyoung Na and Ariana Freitag


'''

import math
import numpy as np
import tensorflow as tf
import keras

#Import keras related stuff
from keras.models import Model
from keras.layers import Input,Activation,Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D

from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import cifar100

from keras import backend as K

# Training parameters
batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 50
num_classes = 100

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3
LR = 0.0001

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return Activation("relu")(BatchNormalization(axis=3)(conv))

    return f

def _bn_relu_conv(**conv_params):

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = Activation("relu")(BatchNormalization(axis=3)(input))
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f
    
def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides, is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    '''
    No bottleneck Residual unit used 
    According to the paper and the diagram, a basic conv block has a 1 -> 3 -> 1 structure.
    '''
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f

class Resnet(object):
    '''
    putting everything together
    '''
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input = Input(shape=input_shape)
        #Conv layer 1
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)

        #Conv layer 2_X
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2
            if (r == 3 or r == 4) :
                block = Dropout(.2)(block)
                


        # Last activation
        block = Activation("relu")(BatchNormalization(axis=3)(block))

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

if __name__ == "__main__":
    '''
    train the model here
    '''
        
    early_stopper = keras.callbacks.callbacks.EarlyStopping(min_delta=0.001, patience=10)
    adam = keras.optimizers.Adam(learning_rate=LR, decay=1e-6,beta_1=0.9, beta_2=0.999, amsgrad=False)

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Softmax to 10 labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Resnet.build((img_channels, img_rows, img_cols), num_classes,basic_block,[3, 4, 6, 3])

    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=epochs,
              validation_split=0.1,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[ early_stopper])

    model.summary()

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

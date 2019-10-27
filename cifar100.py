from __future__ import print_function
import math
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar100
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import MaxPooling2D

'''
Implementation of Resnet-50 from "Deep Residual Learning for Image Recognition" (https://arxiv.org/pdf/1512.03385.pdf)
and "Identity Mappings in Deep Residual Networks" (https://arxiv.org/abs/1603.05027.pdf). More specicifally
our block will look like this https://raw.githubusercontent.com/raghakot/keras-resnet/master/images/residual_block.png

Minyoung Na and Ariana Freitag
'''
# Training parameters
batch_size = 128
epochs = 200
num_classes = 100
depth= 50

# hyperparam
BLCKSIZE = 3 # h and w of block
KEEPPROB =0.9 # probability of activation unit being kept

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

model_type = 'ResNet%d' % (depth)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def drop_block(A):
    '''
    Inputs:
    A is the weights from the activation layer with shape [chan, row, col, batch_size]
    mode turns off drop block during inference

    Return:
    activation layer with correlated blocks dropped

    Implementation from: https://arxiv.org/pdf/1810.12890.pdf
    '''

    block_size = BLCKSIZE
    keep_prob = KEEPPROB

    # compute gamma
    feat_size = A.shape[2]

    term_1 = (1 - keep_prob) / (block_size)**2
    term_2 = (feat_size)**2 / (feat_size-block_size+1)**2
    gamma = term_1 * term_2

    # create mask from Bernoulli distribution
    tfd = tfp.distributions
    dist = tfd.Sample(tfd.Bernoulli(probs=[gamma]), sample_shape=[A.shape[1], A.shape[2],A.shape[3]])
    block_mask = dist.sample()

    # place mask on A
    block_mask = tf.nn.max_pool(block_mask, ksize=(block_size, block_size), strides=(1, 1), padding='SAME')

    # if block_size%2 == 0:
    #     block_mask = block_mask[:, :-1, :-1, :]
    # else:
    #     pass

    drop_mask = tf.squeeze(1-block_mask)

    # apply the mask
    applied_mask = A * drop_mask

    # scale the data
    scaled_data = applied_mask * tf.size(drop_mask)/tf.reduce_sum(drop_mask)

    return scaled_data

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=100):

    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

            if ( stage == 1 or stage == 2 ):
                x = drop_block(x)

        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v2(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              validation_split=0.1,
              callbacks=callbacks)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

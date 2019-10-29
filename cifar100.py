from __future__ import print_function
import math
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, Lambda, MaxPooling2D
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
import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of Resnet-50 from "Deep Residual Learning for Image Recognition" (https://arxiv.org/pdf/1512.03385.pdf)
and "Identity Mappings in Deep Residual Networks" (https://arxiv.org/abs/1603.05027.pdf). More specicifally
our block will look like this https://raw.githubusercontent.com/raghakot/keras-resnet/master/images/residual_block.png

Minyoung Na and Ariana Freitag
"""
# Training parameters
batch_size = 100
epochs = 300
num_classes = 100

# hyperparam
BLCKSIZE = 3  # h and w of block
KEEPPROB = 0.9  # probability of activation unit being kept

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

model_type = "ResNet%d" % (50)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

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
    print("Learning rate: ", lr)
    return lr


def drop_block(A):
    """
    Inputs:
    A is the weights from the activation layer with shape [batch, row, col, chan]

    Return:
    activation layer with correlated blocks dropped

    Implementation from: https://arxiv.org/pdf/1810.12890.pdf
    """
    print('A', A.shape)

    block_size = BLCKSIZE
    keep_prob = KEEPPROB
    feat_size = A.shape[2]
    input_shape = [
        tf.shape(A)[0],
        feat_size - block_size + 1,
        feat_size - block_size + 1,
        A.shape[3],
    ]

    # compute gamma
    term_1 = (1 - keep_prob) / (block_size) ** 2
    term_2 = (feat_size) ** 2 / (feat_size - block_size + 1) ** 2
    gamma = term_1 * term_2

    # set padding
    p1 = (block_size - 1) // 2
    p0 = (block_size - 1) - p1
    padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]

    # create mask from Bernoulli distribution
    block_mask = tf.nn.relu(
        tf.sign(
            gamma - tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)
        )
    )
    block_mask = tf.pad(block_mask, padding)

    # place mask on A
    block_mask = 1 - tf.nn.max_pool(
        block_mask, [1, block_size, block_size, 1], [1, 1, 1, 1], "SAME"
    )

    # img = np.array(block_mask[0,:,:,0])
    # plt.imshow(lol)
    # plt.show()
    # input()

    A = tf.cast(A, dtype=tf.float32)
    block_mask = tf.cast(block_mask, dtype=tf.float32)

    # apply the mask
    applied_mask = tf.multiply(block_mask, A)

    scaled_data = (
        applied_mask
        * tf.cast(tf.size(block_mask), dtype=tf.float32)
        / tf.reduce_sum(block_mask)
    )
    scaled_data = tf.keras.backend.cast(scaled_data, "float32")

    return scaled_data


def drop_block_output(A):
    return [batch_size, A[1], A[2], A[3]]


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):

    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
    )

    x = inputs

    # Either BN-RELU-CONV or CONV-BN-RELU depending on which step you are on
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


def resnet(input_shape):
    # Tried to mimic this architecture from this https://raw.githubusercontent.com/raghakot/keras-resnet/master/images/architecture.png

    # Start model definition.
    num_filters_in = 64

    # Number of resnet blocks in the Resnet-50 
    num_res_blocks = [3, 4, 6, 3]

    inputs = Input(shape=input_shape)

    # v2 performs Conv2D with BN-ReLU on input
    x = resnet_layer(
        inputs=inputs, num_filters=num_filters_in, conv_first=True, strides=2
    )

    # 3x3 maxpool before the residual block starts
    x = MaxPooling2D(
        pool_size=(3, 3), strides=2, padding="same", data_format="channels_last"
    )(x)

    # Instantiate the stack of residual units
    for stage in range(4):
        for block_num in range(num_res_blocks[stage]):
            activation = "relu"
            batch_normalization = True
            strides = 1

            num_filters_out = num_filters_in * 2

            if stage == 0:  # first layer and first stage
                activation = None
                batch_normalization = False

            # bottleneck residual unit , 1 x 3 x 1 structure
            bottleneck = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            bottleneck = resnet_layer(
                inputs=bottleneck, 
                num_filters=num_filters_in, 
                conv_first=False
            )
            bottleneck = resnet_layer(
                inputs=bottleneck, 
                num_filters=num_filters_out, 
                kernel_size=1, 
                conv_first=False
            )

            if block_num == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, bottleneck])

            if stage == 2 or stage == 3:
                layer = Lambda(drop_block, drop_block_output)
                x = layer(x)

        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    input_shape = x_train.shape[1:]

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    print("y_train shape:", y_train.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = resnet(input_shape=input_shape)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=lr_schedule(0)),
        metrics=["accuracy"],
    )
    model.summary()

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(
        factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
    )

    callbacks = [lr_reducer, lr_scheduler]

    #configuration derived from the keras website
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.1,
    )

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        verbose=1,
        workers=4,
        callbacks=callbacks,
    )

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

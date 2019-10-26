import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D


'''
- make sure i have shape of A correct

'''


# hyperparam
BLCKSIZE = 3 # h and w of block
KEEPPROB =0.9 # probability of activation unit being kept


class dropBlock(object):
    def __init__(self):
        self.block_size = BLCKSIZE
        self.keep_prob = KEEPPROB

    def forward(self, A, mode):
        '''
        Inputs:
        A is the weights from the activation layer with shape [batch, row, col, chan]
        mode turns off drop block during inference

        Return:
        activation layer with correlated blocks dropped

        Implementation from: https://arxiv.org/pdf/1810.12890.pdf
        '''

        if mode == 'inference':
            return A
        else:
            # compute gamma
            gamma = self.compute_gamma()

            # create mask from Bernoulli distribution
            mask = tf.distribution.Bernoulli(probs=[gamma]).sample(A.shape[0], A.shape[1:3])

            # place mask on A
            drop_mask = self.compute_block(mask)

            # apply the mask
            applied_mask = A * drop_mask[:,:,None,:]

            # scale the data
            scaled_data = applied_mask * tf.size(drop_mask)/tf.reduce_sum(drop_mask)

            return scaled_data

    def compute_gamma(self, A):
        '''
        Inputs:
        A is the weights from the activation layer [chan, h, w, batch]

        Return:
        single scalar gamma
        '''
        feat_size = A.shape[2]
        term_1 = (1 - self.keep_prob) / (self.block_size)**2
        term_2 = (feat_size)**2 / (feat_size-block_size+1)**2

        return term_1 * term_2

    def compute_block(self, mask):
        block_mask = MaxPooling2D(mask[:,:,None,:],ksize=(self.block_size, self.block_size), strides=(1, 1), padding=self.block_size // 2)

        if self.block_size%2 == 0:
            block_mask = block_mask[:, :-1, :-1, :]
        else:
            pass

        drop_mask = 1-tf.squeeze(block_mask, 1)

        return drop_mask

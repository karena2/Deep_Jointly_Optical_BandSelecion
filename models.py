from keras.layers import Activation
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from functions import *
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.constraints import MinMaxNorm,NonNeg

class My_Band_Selection_method_1(Layer):
    def __init__(self, input_dim=(101, 1), init_value=0.5, parm1=1e-8, bands=25,alpha=3,beta = 0,**kwargs):
        self.input_dim = input_dim
        self.init_value = init_value
        self.my_regularizer = Reg_Binary_0_1_bandas(parm1,bands,alpha,beta)

        super(My_Band_Selection_method_1, self).__init__(**kwargs)

    def get_config(self):  # Cambiar la entrada y compresión
        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'init_value': self.init_value})
        return config

    def build(self, input_shape):
        H_init = np.ones((1, self.input_dim[0], 1,1)) * self.init_value
        H_init = tf.constant_initializer(H_init)
        self.H = self.add_weight(name='H',trainable=True,shape=(1, self.input_dim[0], 1,1), initializer=H_init, regularizer=self.my_regularizer, constraint=MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0))

    def call(self, inputs, **kwargs):
        y = tf.multiply(inputs, self.H)
        return y

def My_network200(pretrained_weights=None, input_size=(101, 1, 1), num_classes=3):
    inputs = Input(input_size)

    conv1 = Conv2D(32, [3, 1], strides=[2, 1], activation='relu')(inputs)
    conv1_1 = Conv2D(32, [3, 1], activation='relu', strides=[2, 1])(conv1)
    conv2 = Conv2D(64, [3, 1], strides=[2, 1], activation='relu')(conv1_1)
    conv3 = Conv2D(128, [3, 1], strides=[2, 1], activation='relu')(conv2)
    conv4 = Conv2D(256, [3, 1], strides=[1, 1], activation='relu')(conv3)
    conv5 = Conv2D(512, [3, 1], strides=[1, 1], activation='relu')(conv4)

    flat = Flatten()(conv5)
    Den1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(flat)
    Den1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(Den1)
    Den1 = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(Den1)
    final = Dense(num_classes, activation='softmax')(Den1)
    model = Model(inputs, final)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def My_networkE2E(pretrained_weights=None, input_size=(101, 1, 1), num_classes=3, bands=25, alpha=3, parm1=1e-8, beta = 0):
    inputs = Input(input_size)
    y=My_Band_Selection_method_1(input_dim=(input_size[0], 1), init_value=1, parm1=parm1, bands=bands, alpha=alpha, beta=beta)(inputs)
    conv1 = Conv2D(32, [3, 1], strides=[2, 1], activation='relu')(y)
    conv1_1 = Conv2D(32, [3, 1], activation='relu', strides=[2, 1])(conv1)
    conv2 = Conv2D(64, [3, 1], strides=[2, 1], activation='relu')(conv1_1)
    conv3 = Conv2D(128, [3, 1], strides=[2, 1], activation='relu')(conv2)
    conv4 = Conv2D(256, [3, 1], strides=[1, 1], activation='relu')(conv3)
    conv5 = Conv2D(512, [3, 1], strides=[1, 1], activation='relu')(conv4)

    flat = Flatten()(conv5)
    Den1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(flat)
    Den1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(Den1)
    Den1 = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(Den1)
    final = Dense(num_classes, activation='softmax')(Den1)
    model = Model(inputs, final)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

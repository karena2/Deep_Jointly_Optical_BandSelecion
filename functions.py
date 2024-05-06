import tensorflow as tf
import random
import numpy as np
from numpy.random import seed
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def round_through(x):
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)


def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.) / 2., 0, 1)


def binary_tanh_unit(x):
    return round_through(hard_sigmoid(x))


class Reg_Binary_0_1_bandas(tf.keras.regularizers.Regularizer):

    def __init__(self, parameter=10, bands=25, alpha=3,
                 beta=0):  # parameter para peso de ambos regularizadores, alpha el valor al que se eleva para mejorar ceros, beta para bajar los ceros

        self.parameter = tf.keras.backend.variable(parameter, name='parameter')
        self.alpha = tf.keras.backend.variable(alpha, name='alpha')
        self.beta = tf.keras.backend.variable(beta, name='beta')
        self.bands = tf.keras.backend.variable(bands, name='bands')

    def __call__(self, x):
        reg_bands = tf.math.pow(tf.reduce_sum(x) - self.bands, 2)
        regularization = self.beta * (
            tf.reduce_sum(tf.multiply(tf.math.pow(tf.pow(x, 2), self.alpha), tf.pow(1 - x, 2)))) + (
                                     1 - self.beta) * tf.reduce_sum(tf.math.pow(tf.pow(x, 2), self.alpha))
        return self.parameter * (regularization + (10) * reg_bands)

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter)),
                'bands': float(tf.keras.backend.get_value(self.bands)),
                'alpha': float(tf.keras.backend.get_value(self.alpha)),
                'beta': float(tf.keras.backend.get_value(self.beta))}



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def split_data(Spectral_data, gt, Portion):
    X_Train = None
    X_Val = None
    for i in range(int(np.max(np.unique(gt)))):  # Se agregó un int 23/11/22

        val = (gt == i + 1)
        val = np.reshape(val, np.shape(Spectral_data)[0])
        firmas = Spectral_data[val]
        Len = firmas.shape[0]
        pp = np.random.permutation(Len)
        if X_Train is not None:

            temporal = firmas[pp[0:int(Len * Portion[0])]]
            X_Train = np.concatenate([X_Train, temporal], axis=0)
            Y_Train = np.concatenate([Y_Train, np.ones((temporal.shape[0], 1)) * i])

            temporal = firmas[pp[int(Len * np.sum(Portion[0])):int(Len * np.sum(Portion[0:2]))]]
            X_Val = np.concatenate([X_Val, temporal], axis=0)
            Y_Val = np.concatenate([Y_Val, np.ones((temporal.shape[0], 1)) * i])

            temporal = firmas[pp[int(Len * np.sum(Portion[0:2])):int(Len * np.sum(Portion[0:3]))]]
            X_Test = np.concatenate([X_Test, temporal], axis=0)
            Y_Test = np.concatenate([Y_Test, np.ones((temporal.shape[0], 1)) * i])

        else:
            X_Train = firmas[pp[0:int(Len * Portion[0])]]
            Y_Train = np.ones((X_Train.shape[0], 1)) * i

            X_Val = firmas[pp[int(Len * np.sum(Portion[0])):int(Len * np.sum(Portion[0:2]))]]
            Y_Val = np.ones((X_Val.shape[0], 1)) * i

            X_Test = firmas[pp[int(Len * np.sum(Portion[0:2])):int(Len * np.sum(Portion[0:3]))]]
            Y_Test = np.ones((X_Test.shape[0], 1)) * i

    p1 = np.random.permutation(len(X_Train))
    p2 = np.random.permutation(len(X_Val))
    p3 = np.random.permutation(len(X_Test))
    X_Train = X_Train[p1]
    Y_Train = Y_Train[p1]
    X_Val = X_Val[p2]
    Y_Val = Y_Val[p2]
    X_Test = X_Test[p3]
    Y_Test = Y_Test[p3]

    Y_Train = to_categorical(Y_Train, int(np.max(np.unique(gt))))
    Y_Val = to_categorical(Y_Val, int(np.max(np.unique(gt))))
    Y_Test = to_categorical(Y_Test, int(np.max(np.unique(gt))))

    return X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test


class Aument_parameters(tf.keras.callbacks.Callback):
    def __init__(self, p_aum, p_step, temp, bandas):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step
        self.tem = temp
        self.bandas = bandas

    def on_epoch_end(self, epoch, logs=None):

        weight = np.asarray(self.model.weights[0])
        weight[weight>=0.5]=1
        weight[weight<0.5]=0
        Nb_best = np.sum(weight)

        if Nb_best == self.bandas:
          self.model.layers[1].my_regularizer.parameter.assign(9999) #Estaba en 9
          self.model.layers[1].my_regularizer.alpha.assign(1)
          self.model.layers[1].my_regularizer.beta.assign(1)
          self.model.layers[1].set_weights(tf.expand_dims(weight,0))

        if (tf.math.floormod(epoch, self.p_step) == (self.p_step - 1)):
            param = self.model.layers[1].my_regularizer.parameter  # Param para valor de los regularizadores
            param = tf.keras.backend.get_value(param)
            param2 = self.model.layers[1].my_regularizer.alpha  # Alpha para gestionar ceros
            param2 = tf.keras.backend.get_value(param2)
            param3 = self.model.layers[1].my_regularizer.beta  # Beta para que baje la funcion
            param3 = tf.keras.backend.get_value(param3)

            if (param >= 10):
                self.model.layers[1].my_regularizer.parameter.assign(999999999)
                self.model.layers[1].my_regularizer.alpha.assign(1)
                self.model.layers[1].my_regularizer.beta.assign(1)
            elif (param3 >= 1):
                self.model.layers[1].my_regularizer.beta.assign(1)
                self.model.layers[1].my_regularizer.parameter.assign(tf.math.pow(param * self.p_aum, tf.constant(0.95)))
            else:
                self.model.layers[1].my_regularizer.parameter.assign(tf.math.pow(param * self.p_aum, tf.constant(0.95)))
                self.model.layers[1].my_regularizer.beta.assign(param3 + (self.tem / 50))

            tf.print('regularizator =' + str(self.model.layers[1].my_regularizer.parameter))
            tf.print('alpha =' + str(self.model.layers[1].my_regularizer.alpha))
            tf.print('beta =' + str(self.model.layers[1].my_regularizer.beta))


class Reload2(tf.keras.callbacks.Callback):
    def __init__(self, save_path, patience=5, save=2 ):
        super(Reload2, self).__init__()
        self.save_path = save_path
        self.patience = patience
        self.best_val_acc = 0
        self.wait = 0
        self.save = save

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs['val_accuracy']
        if epoch > 0:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(self.save_path + 'best_weights.h5')
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = False

    def on_train_end(self, logs=None):
        print('Loading last')
        # self.model.load_weights(self.save_path_last)

class Reload(tf.keras.callbacks.Callback):
    def __init__(self, save_path, patience=5, save=2 ):
        super(Reload, self).__init__()
        self.save_path = save_path
        self.patience = patience
        self.best_val_acc = 0
        self.wait = 0
        self.save = save

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs['val_accuracy']
        if epoch > 150:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(self.save_path + 'best_weights.h5')
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = False

    def on_train_end(self, logs=None):
        print('Loading last')
        # self.model.load_weights(self.save_path_last)

class Inspect_Training(tf.keras.callbacks.Callback):
    def __init__(self, X_Test, Y_Test, Binary, Weight, prevBinary, prevWeight, save_path, seed, N_EPOCHS):
        super().__init__()
        self.X_Test = tf.cast(X_Test, dtype=tf.float32)
        self.Y_Test = Y_Test
        self.Binary = Binary
        self.Weight = Weight
        self.prevBinary = prevBinary
        self.prevWeight = prevWeight
        self.save_path = save_path
        self.seed = seed
        self.N_EPOCHS = N_EPOCHS

    def mydiff(self, l1, l2):
        """
        Returns the numbers of changes per epochs
        """
        r = []
        for i in range(len(l1)):
            r.append(abs(l1[i] - l2[i]))
        return tf.get_static_value(sum(r))

    def on_epoch_end(self, epoch, logs=None):
        # [firma, weight] = self.model(tf.expand_dims(self.X_Test[0], 0), get_H='True')
        weight = np.asarray(self.model.weights[0])
        firma = tf.multiply(tf.expand_dims(tf.expand_dims(self.X_Test[0], 0), -1), weight)
        weight[weight >= 0.5] = 1
        weight[weight < 0.5] = 0
        Nb_best = np.sum(weight)
        print('  Results Best:', Nb_best)
        # Results_Best = self.model.evaluate(self.X_Test, self.Y_Test)
        #if epoch > (self.N_EPOCHS - 2):
        #    plt.figure()
        #    plt.subplot(2, 2, 1), plt.plot(np.squeeze(firma), 'o--', linewidth=0.1, markersize=1), plt.title(
        #        'firma*w-' + str(epoch))
        #    plt.subplot(2, 2, 2), plt.plot(np.squeeze(weight), 'o--', linewidth=0.1, markersize=2), plt.title('Weights')
            # print(np.squeeze(np.asarray([abs(self.Binary[i] - self.Binary[i-1]) for i in range(1, len(self.Binary))])))
        #    print('Selected indices', tf.get_static_value(tf.where(tf.equal(tf.squeeze(firma), 1))))
        #    self.Binary.append(self.mydiff(tf.squeeze(firma), tf.squeeze(self.prevfirma)))
        #    self.Weight.append(self.mydiff(tf.squeeze(weight), tf.squeeze(self.prevWeight)))
        #    plt.subplot(2, 2, 3), plt.plot(self.Binary), plt.title('Binary diff')
        #    plt.subplot(2, 2, 4), plt.plot(self.Weight), plt.title('Weight diff')
        #    plt.savefig(self.save_path + '/Selected_bands' + str(Nb_best) + '_' + str(epoch) + '.png')
            #plt.show()

        self.prevfirma = firma
        self.prevWeight = weight
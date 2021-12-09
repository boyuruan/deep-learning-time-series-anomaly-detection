import numpy as np
from numpy import array, true_divide
import random
from random import randint
import os
# import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, Dropout, MaxPooling1D
# from tensorflow.keras.optimizers_v2 import SGD


def anomaly_detector(prediction_seq, ground_truth_seq):
    dist = np.linalg.norm(ground_truth_seq - prediction_seq)
    if (dist > anm_det_thr):
        return True
    else:
        return False


# History window (number of time stamps taken into account)
w = 2000
# i.e., filter(kernel) size
# Prediction window (number of time stampes required to be
p_w = 300
# predicted)
n_features = 1           # Univariate time series

kernel_size = 2          # Size of filter in conv layers
num_filt_1 = 32          # Number of filters in first conv layer
num_filt_2 = 32          # Number of filters in second conv layer
num_nrn_dl = 40          # Number of neurons in dense layer
num_nrn_ol = p_w         # Number of neurons in output layer

conv_strides = 1
pool_size_1 = 2          # Length of window of pooling layer 1
pool_size_2 = 2          # Length of window of pooling layer 2
pool_strides_1 = 2       # Stride of window of pooling layer 1
pool_strides_2 = 2       # Stride of window of pooling layer 2

epochs = 30
dropout_rate = 0.5       # Dropout rate in the fully connected layer
learning_rate = 2e-5
anm_det_thr = 0.8

df_sine = pd.read_csv(
    'https://raw.githubusercontent.com/swlee23/Deep-Learning-Time-Series-Anomaly-Detection/master/data/sinewave.csv')
# plt.plot(df_sine['sinewave'])
# plt.title('sinewave')
# plt.ylabel('value')
# plt.xlabel('time')
# plt.legend(['sinewave'], loc='upper right')
# plt.figure(figsize=(100, 10))
# plt.show()
# df_sine.head()


def split_sequence(sequence):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + w
        out_end_ix = end_ix + p_w
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
            # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
raw_seq = list(df_sine['sinewave'])

# split into samples
batch_sample, batch_label = split_sequence(raw_seq)
batch_sample = batch_sample.reshape(
    (batch_sample.shape[0], batch_sample.shape[1], n_features))

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv1D(filters=num_filt_1, kernel_size=kernel_size,
          strides=conv_strides, padding='valid', activation='relu', input_shape=(w, n_features)))

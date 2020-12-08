####################################################################################
# Note: 4th.
# Author: Gang-Cheng Huang (Jacky5112)
# Date: Dec.9, 2020
# Lecture: Information Security Training and Education Center, C.C.I.T., N.D.U., Taiwan
####################################################################################

from Case_4_init_data import load_data, load_bin_data
from Case_4_learn_model import MLP_Model
from Case_4_learn_model import CNN_Model
from Case_4_learn_model import DeepConvLSTM_Model

import common
import numpy as np
import pandas as pd

from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Sequential
from keras.layers import Dropout, LeakyReLU
from keras.layers import Dense, Activation, BatchNormalization, Embedding
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# training set
learning_rate = 0.001
validation_split = 0.2
epochs = 150
batch_size = 64 
random_seed = 0
labels = 9

#x_train, y_train, x_test = load_data(save_pickle=True, build_csv=False)
x_train, y_train, x_test = load_bin_data()

minmax_scale = MinMaxScaler(feature_range=(0, 1))
x_train = minmax_scale.fit_transform(x_train)
x_test = minmax_scale.fit_transform(x_test)

y_train = np_utils.to_categorical(y_train)
y_train = np.delete(y_train, np.s_[0], axis=1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=validation_split,
                                                  random_state=random_seed,
                                                  shuffle=True)

#train_history_mlp = MLP_Model(x_train, y_train, x_val, y_val, x_test, labels,
#                                          lr=learning_rate,
#                                          batch_size=batch_size,
#                                          epochs=epochs)

train_history_deepConvLSTM = DeepConvLSTM_Model(x_train, y_train, x_val, y_val, x_test, labels,
                                               lr=learning_rate,
                                               batch_size=batch_size,
                                               epochs=epochs)

#train_history_cnn = CNN_Model(x_train, y_train, x_val, y_val, x_test, labels,
#                                          lr=learning_rate,
#                                          batch_size=batch_size,
#                                          epochs=epochs)


#common.show_train_history(train_history_mlp, 'acc', 'val_acc')
#common.show_train_history(train_history_mlp, 'loss', 'val_loss')

#common.show_train_history(train_history_cnn, 'acc', 'val_acc')
#common.show_train_history(train_history_cnn, 'loss', 'val_loss')

common.show_train_history(train_history_deepConvLSTM, 'accuracy', 'val_accuracy')
common.show_train_history(train_history_deepConvLSTM, 'loss', 'val_loss')
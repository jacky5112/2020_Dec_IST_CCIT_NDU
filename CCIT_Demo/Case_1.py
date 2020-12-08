####################################################################################
# Note: 1st. MNIST
# Author: Gang-Cheng Huang (Jacky5112)
# Date: Dec.9, 2020
# Lecture: Information Security Training and Education Center, C.C.I.T., N.D.U., Taiwan
####################################################################################

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np

from common import show_train_history
from common import show_images_with_labels

# load mnist data 
# x_ prefix -> image
# y_ prefix -> label
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform to 4-demension array
x_train_4d = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test_4d = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalize
x_train_4d_normalize = x_train_4d / 255
x_test_4d_normalize = x_test_4d / 255

# one hot encoding
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onthot = np_utils.to_categorical(y_test)

model = Sequential()

# Conv 1
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))

# Pooling 1
model.add(MaxPool2D(pool_size=(2, 2)))

# Conv 2
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))

# Pooling
model.add(MaxPool2D(pool_size=(2, 2)))

# dropout (to avoid overfitting)
model.add(Dropout(0.25))

# Flattern (7 * 7 -> 36 * 7 * 7 = 1764)
model.add(Flatten())

# hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(10, activation='softmax'))

# summary
print(model.summary())

# define trainning
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# start to train
train_history = model.fit(x = x_train_4d_normalize, y = y_train_onehot, validation_split=0.2, epochs=10, batch_size=300, verbose=2)
## Debug use
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

# evaluate scores
scores = model.evaluate(x_test_4d_normalize, y_test_onthot)

# predict
predtion = model.predict_classes(x_test_4d_normalize)


print(predtion[:20])
show_images_with_labels(x_test, y_test, predtion, index = 0, amount = 10)

####################################################################################
# Note: 6th.
# Author: Gang-Cheng Huang (Jacky5112)
# Date: Dec.9, 2020
# Lecture: Information Security Training and Education Center, C.C.I.T., N.D.U., Taiwan
####################################################################################

import os
os.environ["PATH"] += os.pathsep + 'D:\\Work\\Graphviz2.38\\bin'

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop, Adam
from common import show_train_history
from common import plot_confusion_matrix
from common import show_labels_prob

import numpy as np
import matplotlib.pyplot as plt

# Generate "n_samples" samples from 0 to "n_classes" (step 1)
# Input data with "n_features" features
# f(x + e) != f(x)
n_samples = 4000
n_features = 80
n_classes = 2
e = 0.1
index=[]
labels=[]

for i in range(0, n_classes):
    index.append(i)
    labels.append("label {0}".format(i))

x, y = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=0)
y = to_categorical(y)
x = MinMaxScaler().fit_transform(x) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Build model
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_features,)))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()

# Train model
#train_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, verbose=2)
train_history = model.fit(x, y, epochs=100, batch_size=32, verbose=2)
# Predicit model
y_pred = model.predict_classes(x_test)

# Plot model and Train history
plot_model(model, to_file='case1_model.png', show_shapes=True, show_layer_names=True)
#show_train_history(train_history, 'accuracy', 'val_accuracy')
#show_train_history(train_history, 'loss', 'val_loss')


# 1st x_train and y_train

x0 = x[0]
y0 = y[0]
x0 = np.expand_dims(x0, axis=0)
y0_predict = model.predict(x0)
print(y0_predict)
probability=y0_predict[0]
#show_labels_prob(index, probability)

fig = plt.figure()
plt.bar(index,probability,0.4,color="gray",edgecolor='black')
plt.xlabel("label")
plt.ylabel("probability")
plt.xticks(index, labels)
plt.show()

we_want_to_fake = 0
model_output_layer = model.output
model_input_layer = model.input
cost_function = model_output_layer[0, we_want_to_fake]
gradient_function = K.gradients(cost_function, model_input_layer)[0]
grab_cost_and_gradients_from_model=K.function([model_input_layer,K.learning_phase()], [cost_function, gradient_function] )

cost, gradients = grab_cost_and_gradients_from_model([x0, 0])
n = np.sign(gradients)
x0 += n * e
y0_predict = model.predict(x0)
print(y0_predict)
probability=y0_predict[0]

fig = plt.figure()
plt.bar(index,probability,0.4,color="gray",edgecolor='black')
plt.xlabel("label")
plt.ylabel("probability")
plt.xticks(index, labels)
plt.show()
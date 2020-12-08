####################################################################################
# Note: 2nd. 
# Author: Gang-Cheng Huang (Jacky5112)
# Date: Dec.9, 2020
# Lecture: Information Security Training and Education Center, C.C.I.T., N.D.U., Taiwan
# Dataset. https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip
####################################################################################

import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.applications.vgg16 import VGG16
from keras import backend as K

from common import show_train_history

random_seed = 1234
np.random.seed(random_seed)

# dimensions of our images.
img_width, img_height = 150, 150
img_detail = (img_width, img_height)

# ssd
train_data_dir = 'data_2/train'
validation_data_dir = 'data_2/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16
learning_rate = 0.000001
#momentum = 0.4
#decay = 0.1

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = input_shape)
x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)
x = BatchNormalization()(x)
predictions = Dense(1, activation = 'sigmoid')(x)
model = Model(input = base_model.input, output = predictions)

#model = Sequential([
#    Conv2D(6, kernel_size=(5, 5), input_shape=input_shape, padding='same', activation='relu'),
#    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#    Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu'),
#    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#    Flatten(),
#    Dense(4096, activation='relu'),
#    Dense(4096, activation='relu'),
#    BatchNormalization(),
#    Dense(1, activation='sigmoid')
#    ])

# print summary
print(model.summary())

model.compile(loss='binary_crossentropy',
              #optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=True),
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        dtype='float32',
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        dtype='float32')

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_detail,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='binary',
        seed=random_seed)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=img_detail,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='binary',
        seed=random_seed)

train_history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

## Debug use
common.show_train_history(train_history, 'accuracy', 'val_accuracy')
common.show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate_generator(validation_generator,
                                  nb_validation_samples / batch_size)

print ("Scores: {0}".format(scores[1]))

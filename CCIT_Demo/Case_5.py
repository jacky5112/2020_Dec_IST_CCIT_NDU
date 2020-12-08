####################################################################################
# Note: 5th.
# Author: Gang-Cheng Huang (Jacky5112)
# Date: Dec.9, 2020
# Lecture: Information Security Training and Education Center, C.C.I.T., N.D.U., Taiwan
####################################################################################

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

import common

# init
img_width, img_height, channels = 28, 28, 1
img_shape = (img_width, img_height, channels)
learning_rate = 0.0001
epochs = 50000
batch_size = 32
sample_interval = 500
latent_dim = 100

# ================================================================
# load mnist data
(X_train, _), (_, _) = mnist.load_data()

# rescale from -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))
# ================================================================

# ================================================================
# generator
def BuildGeneratorModel():
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),
        Dense(np.prod(img_shape), activation='tanh'),
        Reshape(img_shape)
        ])

    ## debug use
    print ("generator:")
    print (model.summary())

    noise = Input(shape=(latent_dim,))
    image = model(noise)

    return Model(noise, image)
# ================================================================

# ================================================================
# discriminator
def BuildDiscriminatorModel():
    model = Sequential([
        Flatten(input_shape=img_shape),
        Dense(512),
        LeakyReLU(0.2),
        Dense(256),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
        ])

    ## debug use
    print ("discriminator:")
    print (model.summary())

    image = Input(shape=img_shape)
    validity = model(image)

    return Model(image, validity)
# ================================================================

# ================================================================
# build model
# build and compile the discriminator
discriminator = BuildDiscriminatorModel()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])
discriminator.trainable = False

# build the generator
generator = BuildGeneratorModel()
# generator generates images with noise
z = Input(shape=(latent_dim,))
image = generator(z)

validity = discriminator(image)

# combine model
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', 
                 optimizer=optimizers.Adam(lr=learning_rate))
# ================================================================

# ================================================================
# training
for cnt in range(epochs):
    #-------------------------------------------------------------
    # train discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    images = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # generate a batch of new images
    gen_images = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(images, valid)
    d_loss_fake = discriminator.train_on_batch(gen_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    #-------------------------------------------------------------

    #-------------------------------------------------------------
    # train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, valid)
    #-------------------------------------------------------------

    # save generator images
    if cnt % sample_interval == 0:
        common.sample_images(cnt, generator, latent_dim)
# ================================================================

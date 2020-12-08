from Case_4_init_data import test_path
from Case_4_init_data import img_width, img_height

import common
import numpy as np
import pandas as pd

from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Sequential
from keras.layers import Dropout, LeakyReLU, Conv2D, Conv1D, Flatten, CuDNNLSTM
from keras.layers import Dense, Activation, BatchNormalization, Embedding
from keras.callbacks import EarlyStopping

def _Submission(y_pred, submit_filename):
    ## submit
    read_header = pd.read_csv(test_path, usecols=['Id'])
    read_header = read_header.values

    with open(submit_filename, 'w') as f:
        f.write("Id")

        for i in range(1, 10):
            f.write(",Prediction{0}".format(i))

        f.write("\n")

        for idx_id, val_id in enumerate(read_header):
            f.write(val_id[0])

            for idx_pred, val_pred in enumerate(y_pred[idx_id]):
                f.write(",{0}".format(val_pred))

            f.write("\n")

def CNN_Model(x_train, y_train, x_val, y_val, x_test, labels, lr=0.01, batch_size=20000, epochs=100, model_filename="model_cnn.h5", submit_filename="submission_cnn.csv"):
    print ("-------------------------------------------------------------------------------------\n")
    print("[+] CNN Model.\n")

    x_train = np.reshape(x_train, (x_train.shape[0], img_width, img_height, 1))
    x_val = np.reshape(x_val, (x_val.shape[0], img_width, img_height, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], img_width, img_height, 1))

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same', kernel_initializer='normal', input_shape=(img_width, img_height, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same', kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same', kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', padding='same', kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(labels, activation='softmax', kernel_initializer='normal'))

    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])

    print(model.summary())

    earlystop = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)

    train_history = model.fit(x=x_train, 
                          y=y_train,
                          validation_data=(x_val, y_val), 
                          epochs=epochs, 
                          batch_size=batch_size,
                          callbacks=[earlystop]) 

    model.save(model_filename)
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    _Submission(y_pred, submit_filename)

    return train_history

def MLP_Model(x_train, y_train, x_val, y_val, x_test, labels, lr=0.01, batch_size=20000, epochs=100, model_filename="model_mlp.h5", submit_filename="submission_mlp.csv"):
    print ("-------------------------------------------------------------------------------------\n")
    print("[+] MLP Model.\n")

    model = Sequential()
    model.add(Dense(300, activation='relu', kernel_initializer='normal', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.1))
    model.add(Dense(250, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.1))
    model.add(Dense(150, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.1))
    model.add(Dense(labels, activation='softmax', kernel_initializer='normal'))

    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)

    train_history = model.fit(x=x_train, 
                          y=y_train,
                          validation_data=(x_val, y_val), 
                          epochs=epochs, 
                          batch_size=batch_size,
                          callbacks=[earlystop]) 

    model.save(model_filename)
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    _Submission(y_pred, submit_filename)

    return train_history

def DeepConvLSTM_Model(x_train, y_train, x_val, y_val, x_test, labels, lr=0.01, batch_size=20000, epochs=100, model_filename="model_deepConvLSTM.h5", submit_filename="submission_deepConvLSTM.csv"):
    print ("-------------------------------------------------------------------------------------\n")
    print("[+] DeepConvLSTM Model.\n")

    x_train = np.reshape(x_train, (x_train.shape[0], img_width, img_height))
    x_val = np.reshape(x_val, (x_val.shape[0], img_width, img_height))
    x_test = np.reshape(x_test, (x_test.shape[0], img_width, img_height))

    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    #model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(CuDNNLSTM(256, kernel_initializer='normal'))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    #model.add(Dropout(0.25))
    #model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    #model.add(Dropout(0.25))
    model.add(Dense(labels, activation='softmax', kernel_initializer='normal'))

    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])

    print(model.summary())

    #earlystop = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)

    train_history = model.fit(x=x_train, 
                          y=y_train,
                          validation_data=(x_val, y_val), 
                          epochs=epochs, 
                          batch_size=batch_size)#,
                          #callbacks=[earlystop]) 

    #model.save(model_filename)
    #y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
    #_Submission(y_pred, submit_filename)

    return train_history
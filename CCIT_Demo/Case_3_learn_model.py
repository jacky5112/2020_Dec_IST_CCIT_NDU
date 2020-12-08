from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM, Bidirectional
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers

def LSTM_Model(network_input, n_vocab, lr=0.001):
    """ create the structure of the neural network """

    print ("-------------------------------------------------------------------------------------\n")
    print("[+] LSTM Model.\n")

    model = Sequential()
    model.add(CuDNNLSTM(512, kernel_initializer='normal', input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(CuDNNLSTM(512, kernel_initializer='normal', return_sequences=True))
    model.add(Dropout(0.25))
    model.add(CuDNNLSTM(512, kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_vocab, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr))

    print(model.summary())

    return model

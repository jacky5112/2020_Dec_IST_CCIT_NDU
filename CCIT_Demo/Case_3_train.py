#https://github.com/rileynwong/lstm-jazz/blob/master/train.py
import glob
import pickle
import numpy as np
import os

from common import show_train_history

from music21 import converter, instrument, note, chord

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, TimeDistributed, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def get_notes():
    """
    Convert midi songs to notes. Serialize when done.
    """

    notes = []

    for f in glob.glob('herbie_midi_songs/*.mid'):
        print('Parsing song: ', f)
        midi = converter.parse(f)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: # if file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # notes are flat stucture
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data_3/data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def load_notes():
    """
    Deserialize notes file.
    """
    notes = []

    if os.path.exists('data_3/data/notes') == False:
        notes = get_notes()
    else:
        with open('data_3/data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    print('Preparing sequences...')

    sequence_length = 100

    # Get pitch names
    pitch_names = sorted(set(n for n in notes))

    # Map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, (len(notes) - sequence_length), 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]

        seq_in_int = [note_to_int[char] for char in seq_in]
        network_input.append(seq_in_int)

        seq_out_int = note_to_int[seq_out]
        network_output.append(seq_out_int)

    n_patterns = len(network_input)

    # Reshape for LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize input
    network_input = network_input / float(n_vocab)

    # One-hot encode output
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    print('Creating network...')

    model = Sequential()
    
    model.add(CuDNNLSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
        ))
    model.add(Dropout(0.3)) # Fraction of input units to be dropped during training
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(512))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax')) # Number of possible outputs
    
    """
    model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal', input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=7, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=7, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=7, activation='relu', padding='same', kernel_initializer='normal'))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(CuDNNLSTM(512))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.25))
    model.add(Dense(n_vocab, activation='softmax', kernel_initializer='normal')) # Number of possible outputs
    model.add(Activation('softmax'))
    """
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001))

    return model


def train(model, network_input, network_output):
    """ Train the neural network. """

    print('Training model...')

    filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
    checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
            )
    callbacks_list = [checkpoint]

    train_history = model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

    return train_history


def train_network():
    """ Train! that! network! """
    notes = load_notes()

    # Number of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train_history = train(model, network_input, network_output)

    show_train_history(train_history, 'loss')


if __name__ == '__main__':
    train_network()

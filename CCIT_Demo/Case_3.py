####################################################################################
# Note: 3rd.
# Author: Gang-Cheng Huang (Jacky5112)
# Date: Dec.9, 2020
# Lecture: Information Security Training and Education Center, C.C.I.T., N.D.U., Taiwan
####################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Case_3_init_data import load_data
from Case_3_learn_model import LSTM_Model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, TimeDistributed, MaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from music21 import converter, instrument, note, chord, stream
from keras.callbacks import ModelCheckpoint, History

# init
learning_rate = 0.0001
batch_size = 32
sequence_length = 100
notes_length = 500

def generate_notes(model, notes, network_input, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    pitchnames = sorted(set(item for item in notes))
    
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate "notes_length" notes
    for note_index in range(notes_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
  
def create_midi(prediction_output, filename, notes):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

(notes_data, network_input, network_output, n_vocab) = load_data(False, sequence_length)

# Set up the model
model_lstm = LSTM_Model(network_input, n_vocab, learning_rate)
history = History()

# Fit the model
n_epochs = 200
model_lstm.fit(network_input, network_output, callbacks=[history], epochs=n_epochs, batch_size=128)
model_lstm.save('LSTMmodel.h5')
    
# Use the model to generate a midi
prediction_output = generate_notes(model_lstm, notes_data, network_input, len(set(notes_data)))
create_midi(prediction_output, 'new_back_1', notes_data) 

# Plot the model losses
pd.DataFrame(history.history).plot()
plt.savefig('LSTM_Loss_per_Epoch.png', transparent=True)
plt.close()
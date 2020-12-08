import glob
import pickle
import numpy as np
import os

from music21 import converter, instrument, note, stream, chord
from keras.utils import np_utils

notes_file_path = "data_3/data/notes"
midi_file_path = "data_3/midi_songs/*mid"

def _get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob(midi_file_path):
        midi = converter.parse(file)

        print("[*] Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(notes_file_path, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def _prepare_sequences(notes, n_vocab, sequence_length):
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def load_data(read_save_notes = False, sequence_length=100):
    notes_data = None

    if read_save_notes == True:
        with open(notes_file_path, 'rb') as filepath:
            notes_data = pickle.load(filepath)
    else:
        notes_data = _get_notes()

    # Get the number of pitch names
    n_vocab = len(set(notes_data))

    # Convert notes into numerical input
    network_input, network_output = _prepare_sequences(notes_data, n_vocab, sequence_length)

    return (notes_data, network_input, network_output, n_vocab)
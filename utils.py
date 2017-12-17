import numpy as np

import torch

import string
import unicodedata

import time


# from http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

all_letters = string.ascii_letters + " .,;'-\n"
num_letters = len(all_letters)

def unicode_to_ascii(string):
    return ''.join([
        char for char in unicodedata.normalize('NFD', string) # decomposes unicode characters
        if char in all_letters 
    ])
def read_file(filename):
    return unicode_to_ascii(open(filename, encoding='utf-8').read())

def sample_sequence(source, sequence_length):
    i_start = np.random.randint(len(source) - sequence_length - 1)
    input_sequence = source[i_start:i_start+sequence_length]
    target_sequence = source[i_start+1:i_start+sequence_length+1]
    return input_sequence, target_sequence
def sequence_to_one_hot(sequence):
    tensor = torch.zeros(len(sequence), 1, num_letters)
    for i_letter in range(len(sequence)):
        letter = sequence[i_letter]
        tensor[i_letter][0][all_letters.find(letter)] = 1
    return tensor
def sequence_to_target(sequence):
    tensor = torch.zeros(len(sequence))
    for i_letter in range(len(sequence)):
        letter = sequence[i_letter]
        tensor[i_letter] = all_letters.find(letter)
    return tensor
def one_hot_to_sequence(tensor):
    sequence = ""
    for letter in tensor.squeeze(1):
        sequence += all_letters[np.where(letter.numpy()==1)[0][0]]
    return sequence

def time_since(since):
    now = time.time()
    s = now - since
    m = (s//60)
    s -= m*60
    return '{} minute(s) {} second(s)'.format(int(m), int(s))
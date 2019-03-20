from __future__ import absolute_import, division, print_function

import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *
import tensorflow as tf

sess = tf.InteractiveSession()

path = "data/shakespeare.txt"  #"shakespeare_input.txt"

maxlen = 40

with open(path) as f:
  data = f.read().lower()

X, Y, char_idx = string_to_semi_redundant_sequences(data, seq_maxlen=maxlen, redun_step=2)

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 128)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='rmsprop', loss='categorical_crossentropy',
                       learning_rate=0.01)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=None,
                              checkpoint_path='model')

for i in range(20):
    print('epoch', i+1)
    seed = "shall i compare thee to a summer's day?\n"
    m.fit(X, Y, validation_set=0, batch_size=128,
          n_epoch=1, run_id='shakespeare', shuffle=True)
    print("-- TESTING...")
    print("-- Test with temperature of 1.5 --")
    print(m.generate(600, temperature=1.5, seq_seed=seed))
    print("-- Test with temperature of 0.75 --")
    print(m.generate(600, temperature=0.75, seq_seed=seed))    
    print("-- Test with temperature of 0.25 --")
    print(m.generate(600, temperature=0.25, seq_seed=seed))    

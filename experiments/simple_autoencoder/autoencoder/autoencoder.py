#!/usr/bin/env python

# Very simple autoencoder for 20CR prmsl fields.
# Single, fully-connected layer as encoder+decoder, 32 neurons.
# Very unlikely to work well at all, but this isn't about good
#  results, it's about getting started. 

import os
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset,make_one_shot_iterator
from glob import glob
import numpy

# Set the epochs and steps, because that sets the amount of
#  data we need to provide
n_epochs=100
n_steps=60

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/simple_autoencoder/" %
                   os.getenv('SCRATCH'))
training_files=glob("%s/*.tfd" % input_file_dir)
n_repeat=int(n_epochs*n_steps/len(training_files))+1
train_tfd = tf.constant(training_files)

# Create TensorFlow Dataset objects from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)
tr_data = tr_data.shuffle(buffer_size=10, reshuffle_each_iteration=False)
tr_data = tr_data.repeat(n_repeat)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
# Actually want a tuple of (contents,contents) for source and target
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    ict=tf.reshape(ict,[1,91*180]) # ????
    return (ict,ict)
tr_data = tr_data.map(load_tensor)

# That's set up a Dataset to use - now specify an autoencoder model

# Input placeholder - treat data as 1d
original = tf.keras.layers.Input(shape=(91*180,))
# Encoding layer 32-neuron fully-connected
encoded = tf.keras.layers.Dense(32, activation='relu')(original)
# Output layer - same shape as input
decoded = tf.keras.layers.Dense(91*180, activation='sigmoid')(encoded)

# Model relating original to output
autoencoder = tf.keras.models.Model(original, decoded)
# Choose a loss metric to minimise (RMS)
#  and an optimiser to use (Adadelta)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(x=tr_data, # Get (source,target) pairs from this Dataset
                epochs=n_epochs,
                steps_per_epoch=n_steps,
                shuffle=True,
                verbose=2) # One line per epoch


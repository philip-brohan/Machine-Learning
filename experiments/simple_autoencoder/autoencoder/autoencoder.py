#!/usr/bin/env python

# Very simple autoencoder for 20CR prmsl fields.
# Single, fully-connected layer as encoder+decoder, 32 neurons.
# Very unlikely to work well at all, but this isn't about good
#  results, it's about getting started. 

import os
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
from glob import glob
import numpy

# How many times will we train on each training data point
n_epochs=2

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/simple_autoencoder/training_data" %
                   os.getenv('SCRATCH'))
training_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(training_files)
train_tfd = tf.constant(training_files)

# Create TensorFlow Dataset objects from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)
tr_data = tr_data.shuffle(buffer_size=10, reshuffle_each_iteration=False)
tr_data = tr_data.repeat(n_epochs)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
# Actually want a tuple of (contents,contents) for source and target
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    ict=tf.reshape(ict,[1,91*180]) # ????
    return (ict,ict)
tr_data = tr_data.map(load_tensor)

# We want a similar dataset for testing
test_file_dir=("%s/Machine-Learning-experiments/simple_autoencoder/test_data" %
                   os.getenv('SCRATCH'))
test_files=glob("%s/*.tfd" % test_file_dir)
test_steps=len(test_files)
test_tfd = tf.constant(test_files)
test_data = Dataset.from_tensor_slices(test_tfd)
test_data = test_data.repeat(n_epochs)
test_data = test_data.map(load_tensor)

# That's set up the Datasets to use - now specify an autoencoder model

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
                validation_data=test_data,
                validation_steps=test_steps,
                verbose=2) # One line per epoch

# Save the 

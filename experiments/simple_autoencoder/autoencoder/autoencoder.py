#!/usr/bin/env python

# Very simple autoencoder for 20CR prmsl fields.
# Single, fully-connected layer as encoder+decoder, 32 neurons.
# Very unlikely to work well at all, but this isn't about good
#  results, it's about getting started. 

import os
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.data import Dataset
from glob import glob
import numpy

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/simple_autoencoder/" %
                   os.getenv('SCRATCH'))
train_tfd = tf.constant(glob("%s/*.tfd" % input_file_dir))

# Create TensorFlow Dataset objects from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
# Actually want a tuple of (contents,contents) for source and target
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    return (ict,ict)
tr_data = tr_data.map(load_tensor)

# That's set up a Dataset to use - now specify an autoencoder model

# Input placeholder - treat data as 1d
original = tf.keras.layers.Input(shape=(91,180,))
# Encoding layer 32-neuron fully-connected
encoded = tf.keras.layers.Dense(32, activation='relu')(original)
# Output layer - same shape as input
decoded = tf.keras.layers.Dense(91*180, activation='sigmoid')(encoded)

# Model relating original to output
autoencoder = tf.keras.models.Model(original, decoded)
# Choose a loss metric to minimise (per-pixel binary crossentropy)
#  and an optimiser to use (Adadelta)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x=tr_data, # Get (source,target) pairs from this Dataset
                epochs=2,  # Just enough to see if it works
                steps_per_epoch=10, # ??
                verbose=2) # One line per epoch


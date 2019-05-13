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
import pickle

# How many times will we train on each training data point
n_epochs=100

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/datasets/20CR2c/prmsl/training/" %
                   os.getenv('SCRATCH'))
training_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(training_files)
train_tfd = tf.constant(training_files)


# Create TensorFlow Dataset object from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)

# Use all the data once each epoch
tr_data = tr_data.repeat(n_epochs)

# Present the data in random order
# ?? What does buffer_size do?
tr_data = tr_data.shuffle(buffer_size=10)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    return ict
tr_data = tr_data.map(load_tensor)

# Also need to reshape the data to linear, and produce a tuple
#  (source,target) for model
def to_model(ict):
   ict=tf.reshape(ict,[1,91*180])
   return(ict,ict)
tr_data = tr_data.map(to_model)

# Make similar dataset for testing
test_file_dir=("%s/Machine-Learning-experiments/datasets/20CR2c/prmsl/test/" %
                   os.getenv('SCRATCH'))
test_files=glob("%s/*.tfd" % test_file_dir)
test_steps=len(test_files)
test_tfd = tf.constant(test_files)
test_data = Dataset.from_tensor_slices(test_tfd)
test_data = test_data.repeat(n_epochs)
test_data = test_data.shuffle(buffer_size=10)
test_data = test_data.map(load_tensor)
test_data = test_data.map(to_model)

# That's set up the Datasets to use - now specify an autoencoder model

# Input placeholder - treat data as 1d
original = tf.keras.layers.Input(shape=(91*180,))
# Encoding layer 32-neuron fully-connected
encoded = tf.keras.layers.Dense(32, activation='tanh')(original)
# Output layer - same shape as input
decoded = tf.keras.layers.Dense(91*180, activation='tanh')(encoded)

# Model relating original to output
autoencoder = tf.keras.models.Model(original, decoded)
# Choose a loss metric to minimise (RMS)
#  and an optimiser to use (adadelta)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the autoencoder
history=autoencoder.fit(x=tr_data, # Get (source,target) pairs from this Dataset
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        validation_data=test_data,
                        validation_steps=test_steps,
                        verbose=2) # One line per epoch

# Save the model
save_file="%s/Machine-Learning-experiments/simple_autoencoder/saved_models/Epoch_%04d" % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(autoencoder,save_file)
# Save the training history
history_file="%s/Machine-Learning-experiments/simple_autoencoder/saved_models/history_to_%04d.pkl" % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

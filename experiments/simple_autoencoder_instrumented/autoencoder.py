#!/usr/bin/env python

# Very simple autoencoder for 20CR prmsl fields.
# Single, fully-connected layer as encoder+decoder, 32 neurons.
# Very unlikely to work well at all, but this isn't about good
#  results, it's about getting started. 
#
# This version concentrates on tracking the training of the autoencoder
#  so small batches/epochs and save the state at every point.

import os
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
from glob import glob
import numpy
import pickle

# How much to do between output points
epoch_size=100
# How much training in total
n_epochs=1000

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/datasets/20CR2c/prmsl/training/" %
                   os.getenv('SCRATCH'))
training_files=glob("%s/*.tfd" % input_file_dir)
n_tf=len(training_files)
train_tfd = tf.constant(training_files)

# Create TensorFlow Dataset object from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)

# Repeat the input data enough times that we don't run out 
n_reps=(epoch_size*n_epochs)//n_tf +1
tr_data = tr_data.repeat(n_reps)

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

# Set up a callback to save the model state
checkpoint = ("%s/Machine-Learning-experiments/"+
                "simple_autoencoder_instrumented/"+
                "saved_models/Epoch_{epoch:04d}") % os.getenv('SCRATCH')
if not os.path.isdir(os.path.dirname(checkpoint)):
    os.makedirs(os.path.dirname(checkpoint))
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint, 
                                                 save_weights_only=False,
                                                 verbose=1)

# Train the autoencoder - saving it every epoch
history=autoencoder.fit(x=tr_data, # Get (source,target) pairs from this Dataset
                        epochs=n_epochs,
                        steps_per_epoch=epoch_size,
                        callbacks = [cp_callback],
                        verbose=2) # One line per epoch

# Save the training history
history_file=("%s/Machine-Learning-experiments/"+
              "simple_autoencoder_instrumented/"+
              "saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

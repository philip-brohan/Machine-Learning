#!/usr/bin/env python

# Very simple univariate time-series predictor

# Takes 6-hourly data from 20CR2c at the location of one UK station
#  (from the DWR) and build a predictive model.

import os
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
from glob import glob
import numpy
import pickle
import functools

# How many times will we train on each training data point
n_epochs=20
# How many times in one batch
n_batch=6
# How many timesteps ahead are we forecasting
forecast_steps=1
# How many timesteps back to use making the forecast
window_size=4

# File names for the serialised tensors to train on
input_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                 "DWR/20CR2c/prmsl/training/") %
                   os.getenv('SCRATCH'))
training_files=sorted(glob("%s/1978*.tfd" % input_file_dir))
n_steps=(len(training_files)-forecast_steps)//n_batch

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
# Also, each file contains data for 26 station locations, and
#  here we only want 1, so slice.
@functools.lru_cache(maxsize=None) # Memoize this function
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    ict=tf.slice(ict,[0],[1]) # 1st value only
    ict=tf.reshape(ict,[1,1])
    return ict

# The data are stored as one file per timestep.
# We need to load window_size timesteps and merge them
#  into one tensor.
def load_tensor_window(file_name):
  # Find the set of files forming this window
    idx = tf.where(tf.equal(training_files, file_name))[:,-1]
    slice_files=tf.slice(training_files,idx-window_size+1,[window_size])
    itw=tf.map_fn(load_tensor,slice_files,dtype=numpy.float32)
  # Features as first dim, window size as second (I think)
    itw=tf.reshape(itw,[1,window_size])
    return itw
def load_tensor_window_test(file_name):
  # Find the set of files forming this window
    idx = tf.where(tf.equal(test_files, file_name))[:,-1]
    slice_files=tf.slice(test_files,idx-window_size+1,[window_size])
    itw=tf.map_fn(load_tensor,slice_files,dtype=numpy.float32)
  # Features as first dim, window size as second (I think)
    itw=tf.reshape(itw,[1,window_size])
    return itw

# We need to make both a source and a target dataset
#  source with data from (window_size-1):(len-forecast_steps)
# Each source datset of length window_size
#  target with data from (forecast_steps+window_size-1):len
# Each target dataset of length 1
source_tfd = tf.constant(training_files[(window_size-1):(len(training_files)-forecast_steps)])
source_data = Dataset.from_tensor_slices(source_tfd)
source_data = source_data.repeat(n_epochs)
source_data = source_data.map(load_tensor_window)
target_tfd = tf.constant(training_files[(forecast_steps+window_size-1):(len(training_files))])
target_data = Dataset.from_tensor_slices(target_tfd)
target_data = target_data.repeat(n_epochs)
target_data = target_data.map(load_tensor)
# Zip these together into (source,target) tuples for model fitting.
training_data = Dataset.zip((source_data, target_data))
training_data = training_data.batch(n_batch)

# Repeat the whole process with the test data
test_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
               "DWR/20CR2c/prmsl/test/") %
                   os.getenv('SCRATCH'))
test_files=glob("%s/*.tfd" % test_file_dir)
test_steps=len(test_files)//n_batch
source2_tfd = tf.constant(test_files[(window_size-1):(len(training_files)-forecast_steps)])
source2_data = Dataset.from_tensor_slices(source2_tfd)
source2_data = source2_data.repeat(n_epochs)
source2_data = source2_data.map(load_tensor_window_test)
target2_tfd = tf.constant(test_files[(forecast_steps+window_size-1):(len(training_files))])
target2_data = Dataset.from_tensor_slices(target2_tfd)
target2_data = target2_data.repeat(n_epochs)
target2_data = target2_data.map(load_tensor)
test_data = Dataset.zip((source2_data, target2_data))
test_data = test_data.batch(n_batch)

# That's set up the Datasets to use - now specify an lstm model

# Input placeholder - univariate
original = tf.keras.layers.Input(shape=(1,window_size,))
# Encoding layer 32-neuron LSTM
encoded = tf.keras.layers.LSTM(32,return_sequences=False)(original)
# Output layer - same shape as input
decoded = tf.keras.layers.Dense(1)(encoded)
output = tf.keras.layers.Reshape(target_shape=(1,1,))(decoded)

# Model relating original to output
autoencoder = tf.keras.models.Model(original, output)
# Choose a loss metric to minimise (RMS)
#  and an optimiser to use (adadelta)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the autoencoder
history=autoencoder.fit(x=training_data,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        validation_data=test_data,
                        validation_steps=test_steps,
                        verbose=2) # One line per epoch

# Save the model
save_file=("%s/Machine-Learning-experiments/simple_lstm_predictor/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(autoencoder,save_file)
# Save the training history
history_file=("%s/Machine-Learning-experiments/simple_lstm_predictor/"+
              "saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

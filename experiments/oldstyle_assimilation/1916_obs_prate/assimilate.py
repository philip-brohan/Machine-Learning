#!/usr/bin/env python

# Estimate prmsl fields from (simulated) observations with the coverage of
#  20CRv2c obs in March 1916.
# Use observations of prate.

import os
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
import ML_Utilities
import pickle
import numpy
from glob import glob

# How many epochs to train for
n_epochs=50

def normalise(ic):
    ic = tf.math.sqrt(ic)
    ic *= 100
    return ic

# File names for the serialised tensors to train on
input_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                 "20CR2c/prmsl/training/") %
                   os.getenv('SCRATCH'))
training_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(training_files)
train_tfd = tf.constant(training_files)

# Create TensorFlow Dataset object from the file names
field_data = Dataset.from_tensor_slices(train_tfd)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    ict=tf.reshape(ict,[91,180,1])
    return ict

field_data = field_data.map(load_tensor)
# Use all the data once each epoch
field_data = field_data.repeat(n_epochs)
field_data = field_data.batch(1)

# Same with the test dataset
input_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                 "20CR2c/prmsl/test/") %
                   os.getenv('SCRATCH'))
test_files=glob("%s/*.tfd" % input_file_dir)
test_steps=len(test_files)
test_tfd = tf.constant(test_files)
field_test_data = Dataset.from_tensor_slices(test_tfd)
field_test_data = field_test_data.map(load_tensor)
field_test_data = field_test_data.repeat(n_epochs)
field_test_data = field_test_data.batch(1)

# That's the targets - now make a matching dataset of the 
#   station observations.
# Use the file names from the field files, but repoint them;
#  also adjust to use the training/test split from the field files
def load_observations(file_name):
   # Get the ob tensor file name from the field tensor file name
   file_name=tf.strings.regex_replace(file_name,
                                      'prmsl','prate')
   file_name=tf.strings.regex_replace(file_name,
                                      'datasets/','datasets/obs_1916/')
   file_name=tf.strings.regex_replace(file_name,
                                      'test/','training/')
   file_name=tf.strings.regex_replace(file_name,
                                      'training/2009','test/2009')
   sict=tf.read_file(file_name) # serialised
   ict=tf.parse_tensor(sict,numpy.float32)
   ict=normalise(ict)
   ict=tf.reshape(ict,[488,])
   return ict
obs_data = Dataset.from_tensor_slices(train_tfd)
obs_data = obs_data.repeat(n_epochs)
obs_data = obs_data.map(load_observations)
obs_data = obs_data.batch(1)

# And the test observations
obs_test_data = Dataset.from_tensor_slices(test_tfd)
obs_test_data = obs_test_data.repeat(n_epochs)
obs_test_data = obs_test_data.map(load_observations)
obs_test_data = obs_test_data.batch(1)

# Zip the target and source together for training
training_data = Dataset.zip((obs_data, field_data))
test_data = Dataset.zip((obs_test_data, field_test_data))

# Need to resize data so it's dimensions are a multiple of 8 (3*2-fold pool)
class ResizeLayer(tf.keras.layers.Layer):
   def __init__(self, newsize=None, **kwargs):
      super(ResizeLayer, self).__init__(**kwargs)
      self.resize_newsize = newsize
   def call(self, input):
      return tf.image.resize_images(input, self.resize_newsize,
                                    align_corners=True)
   def get_config(self):
      return {'newsize': self.resize_newsize}


# Build the model:
#   obs->dense network->convolutional network->field

# Input placeholder
original = tf.keras.layers.Input(shape=(488,))
x = tf.keras.layers.Dropout(0.9)(original)

# Obs processing layers
x = tf.keras.layers.Dense(512)(original)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Need to make a 10x20x8 layer for decoding
x = tf.keras.layers.Dense(1600,
    activity_regularizer=tf.keras.regularizers.l1(10e-5))(x)
x = tf.keras.layers.LeakyReLU()(x)
encoded = tf.keras.layers.Reshape(target_shape=(10,20,8,))(x)

# Decoding layers
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(encoded)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), padding='same')(x)
resized=ResizeLayer(newsize=(91,180))(decoded)

# Model relating observations to field
autoencoder = tf.keras.models.Model(original,resized)
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
save_file=("%s/Machine-Learning-experiments/"+
           "oldstyle_assimilation_1916_prate/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(autoencoder,save_file)
history_file=("%s/Machine-Learning-experiments/"+
              "oldstyle_assimilation_1916_prate/"+
              "saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

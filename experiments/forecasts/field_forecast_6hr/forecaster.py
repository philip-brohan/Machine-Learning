#!/usr/bin/env python

# Re-purpose the convolutional autoencoder to do 6-hr forecasts.

import os
import tensorflow as tf
from tensorflow.data import Dataset
import pickle
import numpy
from glob import glob

# How many epochs to train for
n_epochs=30
batch_size=1

# File names for the serialised tensors used as the source
source_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                 "20CR2c/prmsl/training/") %
                   os.getenv('SCRATCH'))
source_files=glob("%s/*.tfd" % source_file_dir)
training_n_steps=len(source_files)//batch_size
source_tfd = tf.constant(source_files)

# Create TensorFlow Dataset object from the file names
source_data = Dataset.from_tensor_slices(source_tfd)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    ict=tf.reshape(ict,[91,180,1])
    return ict

source_data = source_data.map(load_tensor,num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Same but for target tensors (data from 6 hours later)
target_file_dir=(("%s/Machine-Learning-experiments/datasets/f+6hrs/"+
                 "20CR2c/prmsl/training/") %
                   os.getenv('SCRATCH'))
target_files=glob("%s/*.tfd" % target_file_dir)
target_tfd = tf.constant(target_files)
target_data = Dataset.from_tensor_slices(target_tfd)
target_data = target_data.map(load_tensor,num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Combine source and target into one training dataset
training_data = Dataset.zip((source_data, target_data))

# Expand to full length and set batch size
training_data = training_data.repeat(n_epochs)
training_data = training_data.batch(batch_size)
training_data = training_data.prefetch(buffer_size=batch_size)
   
# Do the same but for the test data
test_source_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                       "20CR2c/prmsl/test/") %
                      os.getenv('SCRATCH'))
test_source_files=glob("%s/*.tfd" % test_source_file_dir)
test_n_steps=len(test_source_files)//batch_size
test_source_tfd = tf.constant(test_source_files)
test_source_data = Dataset.from_tensor_slices(test_source_tfd)
test_source_data = test_source_data.map(load_tensor,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_target_file_dir=(("%s/Machine-Learning-experiments/datasets/f+6hrs/"+
                       "20CR2c/prmsl/test/") %
                      os.getenv('SCRATCH'))
test_target_files=glob("%s/*.tfd" % test_target_file_dir)
test_target_tfd = tf.constant(test_target_files)
test_target_data = Dataset.from_tensor_slices(test_target_tfd)
test_target_data = test_target_data.map(load_tensor,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = Dataset.zip((test_source_data, test_target_data))
test_data = test_data.repeat(n_epochs)
test_data = test_data.batch(batch_size)
test_data = test_data.prefetch(buffer_size=batch_size)

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

# Padding and pruning functions for periodic boundary conditions
class LonPadLayer(tf.keras.layers.Layer):
   def __init__(self, index=3, padding=8, **kwargs):
      super(LonPadLayer, self).__init__(**kwargs)
      self.lon_index = index
      self.lon_padding = padding
   def build(self, input_shape):
      self.lon_tile_spec=numpy.repeat(1,len(input_shape))
      self.lon_tile_spec[self.lon_index-1]=3
      self.lon_expansion_slice=[slice(None, None, None)]*len(input_shape)
      self.lon_expansion_slice[self.lon_index-1]=slice(
                                input_shape[self.lon_index-1].value-self.lon_padding,
                                input_shape[self.lon_index-1].value*2+self.lon_padding,
                                None)
      self.lon_expansion_slice=tuple(self.lon_expansion_slice)      
   def call(self, input):
      return tf.tile(input, self.lon_tile_spec)[self.lon_expansion_slice]
   def get_config(self):
      return {'index': self.lon_index}
      return {'padding': self.lon_padding}
class LonPruneLayer(tf.keras.layers.Layer):
   def __init__(self, index=3, padding=8, **kwargs):
      super(LonPruneLayer, self).__init__(**kwargs)
      self.lon_index = index
      self.lon_padding = padding
   def build(self, input_shape):
      self.lon_prune_slice=[slice(None, None, None)]*len(input_shape)
      self.lon_prune_slice[self.lon_index-1]=slice(
                                self.lon_padding,
                                input_shape[self.lon_index-1].value-self.lon_padding,
                                None)
      self.lon_prune_slice=tuple(self.lon_prune_slice)      
   def call(self, input):
     return input[self.lon_prune_slice]
   def get_config(self):
      return {'index': self.lon_index}
      return {'padding': self.lon_padding}

# Input placeholder
original = tf.keras.layers.Input(shape=(91,180,1,))
# Resize to have dimesions divisible by 8
resized = ResizeLayer(newsize=(80,160))(original)
# Wrap-around in longitude for periodic boundary conditions
padded = LonPadLayer(padding=8)(resized)
# Encoding layers
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(padded)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# Forecast layers
x = tf.keras.layers.Dense(600)(encoded)
x = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(600)(x)
forecast = tf.keras.layers.LeakyReLU()(x)
#x = tf.keras.layers.Dropout(0.5)(x)

# Decoding layers
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(forecast)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), padding='same')(x)
# Strip the longitude wrap-around
pruned=LonPruneLayer(padding=8)(decoded)
# Restore to original dimensions
outsize=ResizeLayer(newsize=(91,180))(pruned)

# Model relating original to output
forecaster = tf.keras.models.Model(original,outsize)
# Choose a loss metric to minimise (RMS)
#  and an optimiser to use (adadelta)
forecaster.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the autoencoder
history=forecaster.fit(x=training_data,
                       epochs=n_epochs,
                       steps_per_epoch=training_n_steps,
                       validation_data=test_data,
                       validation_steps=test_n_steps,
                       verbose=2) # One line per epoch

# Save the model
save_file=("%s/Machine-Learning-experiments/"+
           "field_forecast_6hr/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(forecaster,save_file)
history_file=("%s/Machine-Learning-experiments/"+
              "field_forecast_6hr/"+
              "saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

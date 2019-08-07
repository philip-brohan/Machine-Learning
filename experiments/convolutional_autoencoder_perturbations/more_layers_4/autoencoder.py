#!/usr/bin/env python

# Convolutional autoencoder for 20CR prmsl fields.
# This version is all-convolutional - it uses strided convolutions
#  instead of max-pooling, and transpose convolution instead of 
#  upsampling.
# Also this version has 4 layers rather than 3 - for a more 
#  compressed encoded state

import os
import tensorflow as tf
import ML_Utilities
import pickle
import numpy

# How many epochs to train for
n_epochs=5

# Create TensorFlow Dataset object from the prepared training data
(tr_data,n_steps) = ML_Utilities.dataset(purpose='training',
                                         source='20CR2c',
                                         variable='prmsl')
tr_data = tr_data.repeat(n_epochs)

# Also produce a tuple (source,target) for model
def to_model(ict):
   ict=tf.reshape(ict,[91,180,1])
   return(ict,ict)
tr_data = tr_data.map(to_model)
tr_data = tr_data.batch(1)

# Similar dataset from the prepared test data
(tr_test,test_steps) = ML_Utilities.dataset(purpose='test',
                                            source='20CR2c',
                                            variable='prmsl')
tr_test = tr_test.repeat(n_epochs)
tr_test = tr_test.map(to_model)
tr_test = tr_test.batch(1)

# Need to resize data so it's dimensions are a multiple of 32 (5*2-fold pool)
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
# Resize to have dimensions 2^n+1x2^(n+1)+1
resized = ResizeLayer(newsize=(129,249))(original)
# Wrap-around in longitude for periodic boundary conditions
padded = LonPadLayer(padding=8)(resized)
# Encoding layers
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(padded)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
encoded = x

# Decoding layers
x = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides= (2,2), padding='valid')(encoded)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), padding='same')(x)
# Strip the longitude wrap-around
pruned=decoded #LonPruneLayer(padding=8)(decoded)
# Restore to original dimensions
outsize=ResizeLayer(newsize=(91,180))(pruned)

# Model relating original to output
autoencoder = tf.keras.models.Model(original,outsize)
# Choose a loss metric to minimise (RMS)
#  and an optimiser to use (adadelta)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the autoencoder
history=autoencoder.fit(x=tr_data,
                epochs=n_epochs,
                steps_per_epoch=n_steps,
                validation_data=tr_test,
                validation_steps=test_steps,
                verbose=2) # One line per epoch

# Save the model
save_file=("%s/Machine-Learning-experiments/"+
           "convolutional_autoencoder_perturbations/"+
           "more_layers_4/saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(autoencoder,save_file)
history_file=("%s/Machine-Learning-experiments/"+
              "convolutional_autoencoder_perturbations/"+
              "more_layers_4/saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

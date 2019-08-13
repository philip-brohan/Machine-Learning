#!/usr/bin/env python

# Convolutional autoencoder for 20CR prmsl fields.
# This version is all-convolutional - it uses strided convolutions
#  instead of max-pooling, and transpose convolution instead of 
#  upsampling.
# This version uses scaled input fields that have a size (79x159) that
#  match the strided convolution upscaling and downscaling.
# It also works on tensors with a rotated pole - so the data boundary
#  is the equator - this limits the problems with boundary conditions.
#
# This version also compresses the encoded field into a latent space.

import os
import sys
import tensorflow as tf
#tf.enable_eager_execution()
import ML_Utilities
import pickle
import numpy

# Dimensionality of the latent space
latent_dim=100

# How many epochs to train for
n_epochs=50

# Create TensorFlow Dataset object from the prepared training data
(tr_data,n_steps) = ML_Utilities.dataset(purpose='training',
                                         source='rotated_pole/20CR2c',
                                         variable='prmsl')
tr_data = tr_data.repeat(n_epochs)

# Also produce a tuple (source,target) for model
def to_model(ict):
   ict=tf.reshape(ict,[79,159,1])
   return(ict,ict)
tr_data = tr_data.map(to_model)
tr_data = tr_data.batch(1)

# Similar dataset from the prepared test data
(tr_test,test_steps) = ML_Utilities.dataset(purpose='test',
                                            source='rotated_pole/20CR2c',
                                            variable='prmsl')
tr_test = tr_test.repeat(n_epochs)
tr_test = tr_test.map(to_model)
tr_test = tr_test.batch(1)


# Input placeholder
original = tf.keras.layers.Input(shape=(79,159,1,))
# Encoding layers
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(original)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(8, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(4, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2D(2, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)

# N-dimensional latent space representation
x = tf.keras.layers.Reshape(target_shape=(9*19*2,))(x)
encoded = tf.keras.layers.Dense(latent_dim)(x)

# Decoding layers
x = tf.keras.layers.Dense(9*19*2)(encoded)
x = tf.keras.layers.Reshape(target_shape=(9,19,2,))(x)
x = tf.keras.layers.Conv2DTranspose(2, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(4, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(8, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), padding='same')(x)

# Model relating original to output
autoencoder = tf.keras.models.Model(original,decoded)
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
           "latent_space/saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(autoencoder,save_file)
history_file=("%s/Machine-Learning-experiments/"+
              "convolutional_autoencoder_perturbations/"+
              "latent_space/saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

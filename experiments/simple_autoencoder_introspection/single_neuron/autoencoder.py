#!/usr/bin/env python

# Very simple autoencoder for 20CR prmsl fields.
# Single, fully-connected layer as encoder+decoder, 32 neurons.
# Very unlikely to work well at all, but this isn't about good
#  results, it's about getting started. 

# Only one neuron in the hdden layer to simplify introspection.

import os
import tensorflow as tf
import ML_Utilities

# How many epochs to train for
n_epochs=100

# Create TensorFlow Dataset object from the prepared training data
(tr_data,n_steps) = ML_Utilities.dataset(purpose='training',
                                         source='20CR2c',
                                         variable='prmsl')
tr_data = tr_data.repeat(n_epochs)

# Need to reshape the data to linear, and produce a tuple
#  (source,target) for model
def to_model(ict):
   ict=tf.reshape(ict,[1,91*180])
   return(ict,ict)
tr_data = tr_data.map(to_model)

# Similar dataset from the prepared test data
(tr_test,test_steps) = ML_Utilities.dataset(purpose='test',
                                            source='20CR2c',
                                            variable='prmsl')
tr_test = tr_test.repeat(n_epochs)
tr_test = tr_test.map(to_model)

# Input placeholder - treat data as 1d
original = tf.keras.layers.Input(shape=(91*180,))
# Encoding layer: fully-connected
encoded = tf.keras.layers.Dense(1, activation='tanh')(original)
# Output layer - same shape as input
decoded = tf.keras.layers.Dense(91*180, activation='tanh')(encoded)

# Model relating original to output
autoencoder = tf.keras.models.Model(original, decoded)
# Choose a loss metric to minimise (RMS)
#  and an optimiser to use (adadelta)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(x=tr_data, # Get (source,target) pairs from this Dataset
                epochs=n_epochs,
                steps_per_epoch=n_steps,
                validation_data=tr_test,
                validation_steps=test_steps,
                verbose=2) # One line per epoch

# Save the model
save_file=("%s/Machine-Learning-experiments/"+
           "simple_autoencoder_introspection/single_neuron/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(autoencoder,save_file)

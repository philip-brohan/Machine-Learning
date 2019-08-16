#!/usr/bin/env python

# Variational autoencoder for 20CR prmsl fields.

# This should be a generative model.

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
n_epochs=500

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

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

# Input placeholder
original = tf.keras.layers.Input(shape=(79,159,1,), name='encoder_input')
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
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

# Use reparameterization trick to push the sampling out as input
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Define an encoder model
encoder = tf.keras.models.Model(original, [z_mean, z_log_var, z], name='encoder')

# Decoding layers
encoded=tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input') # Will be 'z' above
x = tf.keras.layers.Dense(9*19*2)(encoded)
x = tf.keras.layers.Reshape(target_shape=(9,19,2,))(x)
x = tf.keras.layers.Conv2DTranspose(2, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(4, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Conv2DTranspose(8, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.LeakyReLU()(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), padding='same')(x) # Will be 75x159 - same as input

# Define a generator (decoder) model
generator = tf.keras.models.Model(encoded, decoded, name='generator')

# Combine the encoder and the generator into an autoencoder
#autoInput = tf.keras.layers.Input(shape=(79,159,1,), name='autoencoder_input')
output = generator(encoder(original)[2])
autoencoder = tf.keras.models.Model(inputs=original, outputs=output, name='autoencoder')

# Specify a loss function - combination of the reconstruction loss
#  and the KL divergence of the latent space (from a multivariate gaussian).

reconstruction_loss = tf.keras.losses.mse(original, output)
reconstruction_loss *= latent_dim * 100

kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    
autoencoder.add_loss(vae_loss)

autoencoder.compile(optimizer='adadelta')

# Train the autoencoder
history=autoencoder.fit(x=tr_data,
                epochs=n_epochs,
                steps_per_epoch=n_steps,
                validation_data=tr_test,
                validation_steps=test_steps,
                verbose=2) # One line per epoch

# Save the models
save_dir=("%s/Machine-Learning-experiments/"+
           "variational_autoencoder/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),n_epochs)
if not os.path.isdir(os.path.dirname(save_dir)):
    os.makedirs(os.path.dirname(save_dir))
for model in ['autoencoder','encoder','gererator']:
    if not os.path.isdir(os.path.dirname("%s/%s" % (save_dir,model))):
        os.makedirs(os.path.dirname("%s/%s" % (save_dir,model)))
tf.keras.models.save_model(autoencoder,"%s/autoencoder" % save_dir)
tf.keras.models.save_model(encoder,"%s/encoder" % save_dir)
tf.keras.models.save_model(generator,"%s/generator" % save_dir)
history_file=("%s/Machine-Learning-experiments/"+
              "variational_autoencoder/"+
              "saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),n_epochs)
pickle.dump(history.history, open(history_file, "wb"))

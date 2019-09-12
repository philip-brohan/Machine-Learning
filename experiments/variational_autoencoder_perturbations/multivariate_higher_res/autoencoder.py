#!/usr/bin/env python

# Variational autoencoder for 20CR prmsl fields.
# This version does pressure, temperature, and z500.

# This should be a generative model.

import os
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
import pickle
import numpy
from glob import glob

# How many epochs to train for
n_epochs=100

# Dimensionality of the latent space
latent_dim=100

# Target data setup
buffer_size=100
batch_size=1

# Training datasets with three variables
input_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                 "rotated_pole_high-res/" +
                "20CR2c/prmsl/training/") %
                   os.getenv('SCRATCH'))
prmsl_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(prmsl_files)
tr_tfd = tf.constant(prmsl_files)

# Create TensorFlow Dataset object from the file names
tr_data = Dataset.from_tensor_slices(tr_tfd).repeat(n_epochs)

# From the prmsl file name, make a three-variable tensor
def load_tensor(file_name):
    sict  = tf.read_file(file_name)
    prmsl = tf.parse_tensor(sict,numpy.float32)
    prmsl = tf.reshape(prmsl,[159,319,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'prmsl','air.2m')
    sict  = tf.read_file(file_name)
    t2m   = tf.parse_tensor(sict,numpy.float32)
    t2m   = tf.reshape(t2m,[159,319,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'air.2m','z500')
    sict  = tf.read_file(file_name)
    prate = tf.parse_tensor(sict,numpy.float32)
    prate = tf.reshape(prate,[159,319,1])
    ict = tf.concat([prmsl,t2m,prate],2) # Now [159,319,3]
    ict = tf.reshape(ict,[159,319,3])
    return ict

tr_data = tr_data.map(load_tensor)
tr_data = tr_data.shuffle(buffer_size).batch(batch_size)
tr_data = Dataset.zip((tr_data, tr_data))

# Same for the test dataset
input_file_dir=(("%s/Machine-Learning-experiments/datasets/"+
                 "rotated_pole_high-res/" +
                "20CR2c/prmsl/test/") %
                   os.getenv('SCRATCH'))
prmsl_files=glob("%s/*.tfd" % input_file_dir)
test_steps=len(prmsl_files)
test_tfd = tf.constant(prmsl_files)
test_data = Dataset.from_tensor_slices(test_tfd).repeat(n_epochs)
test_data = test_data.map(load_tensor)
test_data = test_data.batch(batch_size)
test_data = Dataset.zip((test_data, test_data))

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
original = tf.keras.layers.Input(shape=(159,319,3,), name='encoder_input')
# Encoding layers
x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(original)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(3, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(9, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(27, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(81, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# N-dimensional latent space representation
x = tf.keras.layers.Reshape(target_shape=(9*19*81,))(x)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

# Use reparameterization trick to push the sampling out as input
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Define an encoder model
encoder = tf.keras.models.Model(original, [z_mean, z_log_var, z], name='encoder')

# Decoding layers
encoded=tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input') # Will be 'z' above
x = tf.keras.layers.Dense(9*19*81)(encoded)
x = tf.keras.layers.Reshape(target_shape=(9,19,81,))(x)
x = tf.keras.layers.Conv2DTranspose(81, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(27, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(9, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(3, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
decoded = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x) # Will be 159x319x3 - same as input

# Define a generator (decoder) model
generator = tf.keras.models.Model(encoded, decoded, name='generator')

# Combine the encoder and the generator into an autoencoder
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

# Save model and history state after every epoch
history={}
history['loss']=[]
history['val_loss']=[]
class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir=("%s/Machine-Learning-experiments/"+
                   "variational_autoencoder_perturbations/"+
                   "multivariate_higher_res/saved_models/Epoch_%04d") % (
                         os.getenv('SCRATCH'),epoch)
        if not os.path.isdir(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        for model in ['autoencoder','encoder','gererator']:
            if not os.path.isdir(os.path.dirname("%s/%s" % (save_dir,model))):
                os.makedirs(os.path.dirname("%s/%s" % (save_dir,model)))
        tf.keras.models.save_model(autoencoder,"%s/autoencoder" % save_dir)
        tf.keras.models.save_model(encoder,"%s/encoder" % save_dir)
        tf.keras.models.save_model(generator,"%s/generator" % save_dir)
        history['loss'].append(logs['loss'])
        history['val_loss'].append(logs['val_loss'])
        history_file=("%s/Machine-Learning-experiments/"+
                      "variational_autoencoder_perturbations/"+
                      "multivariate_higher_res/saved_models/"+
                      "history_to_%04d.pkl") % (
                         os.getenv('SCRATCH'),epoch)
        pickle.dump(history, open(history_file, "wb"))
        
saver = CustomSaver()

# Train the autoencoder
history=autoencoder.fit(x=tr_data,
                epochs=n_epochs,
                steps_per_epoch=n_steps,
                validation_data=test_data,
                validation_steps=test_steps,
                verbose=2,
                callbacks=[saver])


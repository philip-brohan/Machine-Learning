#!/usr/bin/env python

# Convolutional autoencoder for 20CR prmsl fields.
# This version does wind, temperature, and prmsl.

import os
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
import pickle
import numpy
from glob import glob

# How many epochs to train for
n_epochs=10

# How big a latent space
latent_dim=100

# Target data setup
buffer_size=100
batch_size=1

# Training datasets with four variables
input_file_dir=(("%s/Machine-Learning-experiments/datasets/uk_centred/" +
                "20CR2c/air.2m/training/") %
                   os.getenv('SCRATCH'))
t2m_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(t2m_files)
tr_tfd = tf.constant(t2m_files)

# Create TensorFlow Dataset object from the file names
tr_data = Dataset.from_tensor_slices(tr_tfd).repeat(n_epochs)

# From the t2m file name, make a four-variable tensor
def load_tensor(file_name):
    sict  = tf.read_file(file_name)
    t2m = tf.parse_tensor(sict,numpy.float32)
    t2m = tf.reshape(t2m,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'air.2m','prmsl')
    sict  = tf.read_file(file_name)
    prmsl = tf.parse_tensor(sict,numpy.float32)
    prmsl = tf.reshape(prmsl,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'prmsl','uwnd.10m')
    sict  = tf.read_file(file_name)
    uwnd  = tf.parse_tensor(sict,numpy.float32)
    uwnd  = tf.reshape(uwnd,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'uwnd.10m','vwnd.10m')
    sict  = tf.read_file(file_name)
    vwnd  = tf.parse_tensor(sict,numpy.float32)
    vwnd  = tf.reshape(vwnd,[79,159,1])
    ict = tf.concat([t2m,prmsl,uwnd,vwnd],2) # Now [79,159,4]
    ict = tf.reshape(ict,[79,159,4])
    return ict

tr_data = tr_data.map(load_tensor)
tr_data = tr_data.shuffle(buffer_size).batch(batch_size)
tr_data = Dataset.zip((tr_data, tr_data))

# Same for the test dataset
input_file_dir=(("%s/Machine-Learning-experiments/datasets/uk_centred/" +
                "20CR2c/air.2m/test/") %
                   os.getenv('SCRATCH'))
t2m_files=glob("%s/*.tfd" % input_file_dir)
test_steps=len(t2m_files)
test_tfd = tf.constant(t2m_files)
test_data = Dataset.from_tensor_slices(test_tfd).repeat(n_epochs)
test_data = test_data.map(load_tensor)
test_data = test_data.batch(batch_size)
test_data = Dataset.zip((test_data, test_data))

# Input placeholder
original = tf.keras.layers.Input(shape=(79,159,4,), name='encoder_input')
# Encoding layers
x = tf.keras.layers.Conv2D(4, (3, 3), padding='same')(original)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(12, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(36, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(108, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
# N-dimensional latent space representation
x = tf.keras.layers.Reshape(target_shape=(9*19*108,))(x)
encoded = tf.keras.layers.Dense(latent_dim,)(x)

# Define an encoder model
encoder = tf.keras.models.Model(original, encoded, name='encoder')

# Decoding layers
encoded = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
x = tf.keras.layers.Dense(9*19*108,)(encoded)
x = tf.keras.layers.Reshape(target_shape=(9,19,108,))(x)
x = tf.keras.layers.Conv2DTranspose(108, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(36, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(12, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
decoded = tf.keras.layers.Conv2D(4, (3, 3), padding='same')(x) # Will be 75x159x4 - same as input

# Define a generator (decoder) model
generator = tf.keras.models.Model(encoded, decoded, name='generator')

# Combine the encoder and the generator into an autoencoder
output = generator(encoder(original))
autoencoder = tf.keras.models.Model(inputs=original, outputs=output, name='autoencoder')

reconstruction_loss = tf.keras.losses.mse(original, output)

autoencoder.add_loss(reconstruction_loss)

autoencoder.compile(optimizer='adadelta')

# Save model and history state after every epoch
history={}
history['loss']=[]
history['val_loss']=[]
class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir=("%s/Machine-Learning-experiments/"+
                   "convolutional_autoencoder_perturbations/"+
                   "multivariate_uk_centred/saved_models/Epoch_%04d") % (
                         os.getenv('SCRATCH'),epoch)
        if not os.path.isdir(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        for model in ['autoencoder','encoder','generator']:
            if not os.path.isdir(os.path.dirname("%s/%s" % (save_dir,model))):
                os.makedirs(os.path.dirname("%s/%s" % (save_dir,model)))
        tf.keras.models.save_model(autoencoder,"%s/autoencoder" % save_dir)
        tf.keras.models.save_model(encoder,"%s/encoder" % save_dir)
        tf.keras.models.save_model(generator,"%s/generator" % save_dir)
        history['loss'].append(logs['loss'])
        history['val_loss'].append(logs['val_loss'])
        history_file=("%s/Machine-Learning-experiments/"+
                      "convolutional_autoencoder_perturbations/"+
                      "multivariate_uk_centred/saved_models/history_to_%04d.pkl") % (
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


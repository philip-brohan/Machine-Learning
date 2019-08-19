#!/usr/bin/env python

# Deep Convolutional Generational Adversarial Network
#  Generator trains to produce a surface weather field
#   (temperature, mslp, prate) from a random seed.
# Based on Radford et al. 2016, and 
#  https://www.tensorflow.org/beta/tutorials/generative/dcgan

import os
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.data import Dataset
from glob import glob
import numpy
import time

# How many epochs to train for
n_epochs=500

# Target data setup
buffer_size=100
batch_size=1

# Latent space (noise) dimensionality
latent_dim=100

# Target tensors are the same as those used in the variational autoencoder,
#  except that they are multivariate.

input_file_dir=(("%s/Machine-Learning-experiments/datasets/rotated_pole/" +
                "20CR2c/prmsl/training/") %
                   os.getenv('SCRATCH'))
prmsl_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(prmsl_files)
target_tfd = tf.constant(prmsl_files)

# Create TensorFlow Dataset object from the file names
target_data = Dataset.from_tensor_slices(target_tfd)

# From the prmsl file name, make a three-variable tensor
def load_tensor(file_name):
    sict  = tf.read_file(file_name)
    prmsl = tf.parse_tensor(sict,numpy.float32)
    prmsl = tf.reshape(prmsl,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'prmsl','air.2m')
    sict  = tf.read_file(file_name)
    t2m   = tf.parse_tensor(sict,numpy.float32)
    t2m   = tf.reshape(t2m,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'air.2m','prate')
    sict  = tf.read_file(file_name)
    prate = tf.parse_tensor(sict,numpy.float32)
    prate = tf.reshape(prate,[79,159,1])
    ict = tf.concat([prmsl,t2m,prmsl],2) # Now [79,159,3]
    ict = tf.reshape(ict,[79,159,3])
    return ict

target_data = target_data.map(load_tensor)
target_data = target_data.shuffle(buffer_size).batch(batch_size)

# Define the GAN - two models.

# Define a discriminator model, that takes [79,159,3] tensors
#  and classifies them as real (+ve) or fake (-ve)

def make_discriminator_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(27, (3, 3), strides=(1, 1), padding='same',
                                               input_shape=[79, 159, 3]))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(9, (3, 3), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(3, (3, 3), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(1, (3, 3), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

discriminator = make_discriminator_model()
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

#discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# The discriminator is doing well if it can distinguish real fields from
#  generated fakes. It needs a low false positive rate and a low
#  false negative rate.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define a generator model, that makes [79,159,3] tensors
#  from a random seed. Based on the variational autoencoder.

def make_generator_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(9*19*81, input_shape=(latent_dim,)))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((9, 19, 81)))
    assert model.output_shape == (None, 9, 19, 81) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(81, (3, 3), strides=(2, 2), padding='valid'))
    assert model.output_shape == (None, 19, 39, 81)
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(27, (3, 3), strides=(2, 2), padding='valid'))
    assert model.output_shape == (None, 39, 79, 27)
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(9, (3, 3), strides=(2, 2), padding='valid'))
    assert model.output_shape == (None, 79, 159, 9)
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same'))
    assert model.output_shape == (None, 79, 159, 3)

    return model

generator = make_generator_model()
generator_optimizer = tf.train.AdamOptimizer(1e-4)
#generator_optimizer = tf.keras.optimizers.Adam(1e-4)

# The generator is doing well if it's fooling the discriminator
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Specify what to do every training step
#  Pass in a batch of real fields
#  Make a noise sample
#  Generate a batch of fake images from the noise
#  Run the discriminator on both the real and fake images
#  Calculate discriminator loss and generator loss
#  Update the generator and the discriminator from their loss gradients

#@tf.function # Makes this function work in tf graphs
def train_step(fields):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(fields, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Save the models
def save_models(epoch):
    save_dir=("%s/Machine-Learning-experiments/"+
               "DCGAN/"+
               "saved_models/Epoch_%04d") % (
                     os.getenv('SCRATCH'),epoch)
    if not os.path.isdir(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    for model in ['discriminator','generator']:
        if not os.path.isdir(os.path.dirname("%s/%s" % (save_dir,model))):
            os.makedirs(os.path.dirname("%s/%s" % (save_dir,model)))
    tf.keras.models.save_model(discriminator,"%s/discriminator" % save_dir)
    tf.keras.models.save_model(generator,"%s/generator" % save_dir)
    
# Train the models
for epoch in range(n_epochs):
    start = time.time()

    for field_batch in target_data:
      train_step(field_batch)

    save_models(epoch)
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


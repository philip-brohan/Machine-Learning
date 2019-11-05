#!/usr/bin/env python

# Get annual and diurnal cycle position from 
#  wind, temperature, and prmsl.

import os
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
import pickle
import numpy
from glob import glob

# How many epochs to train for
n_epochs=11

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

# From the t2m file name, calculate the cycle locations
mday=tf.constant([0,31,59,90,120,151,181,212,243,273,304,334])
def cycle_tensor(file_name):
    fdte = tf.strings.substr(file_name,-17,17)
    year = tf.strings.substr(fdte,0,5)
    mnth = tf.strings.substr(fdte,5,2)
    day  = tf.strings.substr(fdte,8,2)
    day  = tf.cond(tf.math.equal(mnth+day,'0229'),
                               lambda: tf.constant('28'),lambda: day)
    hour = tf.strings.substr(fdte,11,2)
    diurnal = tf.strings.to_number(hour)/24.0
    mnth_i  = tf.strings.to_number(mnth,tf.dtypes.int32)-1
    annual  = (tf.cast(tf.gather(mday,mnth_i),tf.dtypes.float32)+
               tf.strings.to_number(day))*3.141592/365.0
    annual  = tf.math.sin(annual)
    cycles = tf.stack([annual,diurnal], 0)
    return cycles

tr_source = tr_data.map(load_tensor)
tr_target = tr_data.map(cycle_tensor)
tr_data = Dataset.zip((tr_source, tr_target))
tr_data = tr_data.shuffle(buffer_size).batch(batch_size)

# Same for the test dataset
input_file_dir=(("%s/Machine-Learning-experiments/datasets/uk_centred/" +
                "20CR2c/air.2m/test/") %
                   os.getenv('SCRATCH'))
t2m_files=glob("%s/*.tfd" % input_file_dir)
test_steps=len(t2m_files)
test_tfd = tf.constant(t2m_files)
test_data = Dataset.from_tensor_slices(test_tfd).repeat(n_epochs)
test_source = test_data.map(load_tensor)
test_target = test_data.map(cycle_tensor)
test_data = Dataset.zip((test_source, test_target))
test_data = test_data.batch(batch_size)

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
x = tf.keras.layers.Reshape(target_shape=(9*19*108,))(x)
encoded = tf.keras.layers.Dense(2,)(x)

encoder = tf.keras.models.Model(inputs=original, 
                                outputs=encoded, 
                                name='encoder')

encoder.compile(optimizer='adadelta',loss='mean_squared_error')

# Save model and history state after every epoch
history={}
history['loss']=[]
history['val_loss']=[]
class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir=("%s/Machine-Learning-experiments/"+
                   "convolutional_autoencoder_perturbations/"+
                   "check_cycles/saved_models/Epoch_%04d") % (
                         os.getenv('SCRATCH'),epoch)
        if not os.path.isdir(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        for model in ['encoder']:
            if not os.path.isdir(os.path.dirname("%s/%s" % (save_dir,model))):
                os.makedirs(os.path.dirname("%s/%s" % (save_dir,model)))
        tf.keras.models.save_model(encoder,"%s/encoder" % save_dir)
        history['loss'].append(logs['loss'])
        history['val_loss'].append(logs['val_loss'])
        history_file=("%s/Machine-Learning-experiments/"+
                      "convolutional_autoencoder_perturbations/"+
                      "check_cycles/saved_models/history_to_%04d.pkl") % (
                         os.getenv('SCRATCH'),epoch)
        pickle.dump(history, open(history_file, "wb"))
        
saver = CustomSaver()

# Train the encoder
history=encoder.fit(x=tr_data,
                epochs=n_epochs,
                steps_per_epoch=n_steps,
                validation_data=test_data,
                validation_steps=test_steps,
                verbose=1,
                callbacks=[saver])


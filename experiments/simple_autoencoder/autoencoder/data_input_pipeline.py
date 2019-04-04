#!/usr/bin/env python

# Test of the tf data input pipeline

import os
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.data import Dataset,make_one_shot_iterator
from glob import glob
import numpy

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/simple_autoencoder/" %
                   os.getenv('SCRATCH'))
train_tfd = tf.constant(glob("%s/*.tfd" % input_file_dir))

# Create TensorFlow Dataset objects from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    return ict
tr_data = tr_data.map(load_tensor)

# tr_data is now a useable dataset providing the training data
#  iterate over it to show it's working.
fn_iterator = make_one_shot_iterator(tr_data)
next_fn = fn_iterator.get_next()

while True:
    elem = next_fn
    print(elem)

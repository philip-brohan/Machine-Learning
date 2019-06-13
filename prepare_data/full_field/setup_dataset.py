#!/usr/bin/env python

# Demonstrate construction of a tf.data.Dataset from data files

import os
import tensorflow as tf
from tensorflow.data import Dataset
tf.enable_eager_execution()
from glob import glob
import numpy

# File names for the serialised tensors to train on
input_file_dir=("%s/Machine-Learning-experiments/datasets/20CR2c/prmsl/training/" %
                   os.getenv('SCRATCH'))
train_tfd = tf.constant(glob("%s/*.tfd" % input_file_dir))

# Create TensorFlow Dataset object from the file names
tr_data = Dataset.from_tensor_slices(train_tfd)

# The Dataset will run out of data when it has used each file once
#  extend it to 10,000 samples by repeating it
tr_data = tr_data.repeat(10000//len(train_tfd)+1)

# Present the data in random order
# ?? What does buffer_size do?
tr_data = tr_data.shuffle(buffer_size=10)

# We don't want the file names, we want their contents, so
#  add a map to convert from names to contents.
def load_tensor(file_name):
    sict=tf.read_file(file_name) # serialised
    ict=tf.parse_tensor(sict,numpy.float32)
    return ict
tr_data = tr_data.map(load_tensor)

# tr_data is now ready for use

# Note that you may need to add another .map to get the data into the right shape and format
#  for your model - some require pairs of tensors (source,target), for example.

# To see that it's working let's extract and print the data from the dataset
from tensorflow.data import make_one_shot_iterator
from tensorflow.python.framework.errors_impl import OutOfRangeError

fn_iterator = make_one_shot_iterator(tr_data)

while True:
    try:
        elem = fn_iterator.get_next()
        print(elem)
    except OutOfRangeError:
        print("Out of data")
        break

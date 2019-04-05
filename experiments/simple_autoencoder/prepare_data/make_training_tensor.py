#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

import tensorflow as tf
tf.enable_eager_execution()
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os

# Get the datetime to plot from commandline arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year",
                    type=int,required=True)
parser.add_argument("--month", help="Integer month",
                    type=int,required=True)
parser.add_argument("--day", help="Day of month",
                    type=int,required=True)
parser.add_argument("--hour", help="Hour of day (0 to 23)",
                    type=int,required=True)
parser.add_argument("--member", help="Ensemble member",
                    default=1,type=int,required=False)
parser.add_argument("--version", help="20CR version",
                    default='2c',type=str,required=False)
parser.add_argument("--variable", help="20CR variable",
                    default='prmsl',type=str,required=False)
parser.add_argument("--test", help="test data, not training",
                    action="store_true")
parser.add_argument("--opfile", help="tf data file name",
                    default=None,
                    type=str,required=False)
args = parser.parse_args()
if args.opfile is None:
    group='training_data'
    if args.test: group='test_data'
    opfs="%s/Machine-Learning-experiments/simple_autoencoder/"+\
          "%s/%04d-%02d-%02d:%02d_%s.tfd"
    args.opfile = opfs % (os.getenv('SCRATCH'),group,args.year,
                          args.month,args.day,args.hour,args.variable)

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

ic=twcr.load(args.variable,datetime.datetime(args.year,args.month,
                                            args.day,args.hour),
             version=args.version)

# Reduce to selected ensemble member
ic=ic.extract(iris.Constraint(member=args.member))

# Convert to Tensor
ict=tf.convert_to_tensor(ic.data, numpy.float32)

# Write to tfrecord file
sict=tf.serialize_tensor(ict)
tf.write_file(args.opfile,sict)

#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

# Do it with a 6hr offset - to make targets for forecasting

import tensorflow as tf
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os

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
parser.add_argument("--source", help="Data source",
                    default='20CR2c',type=str,required=False)
parser.add_argument("--variable", help="variable name",
                    default='prmsl',type=str,required=False)
parser.add_argument("--test", help="test data, not training",
                    action="store_true")
parser.add_argument("--opfile", help="tf data file name",
                    default=None,
                    type=str,required=False)
args = parser.parse_args()
if args.opfile is None:
    purpose='training'
    if args.test: purpose='test'
    args.opfile=("%s/Machine-Learning-experiments/datasets/f+6hrs/%s/%s/%s/%04d-%02d-%02d:%02d.tfd" %
                       (os.getenv('SCRATCH'),args.source,args.variable,purpose,
                        args.year,args.month,args.day,args.hour))

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

if args.source=='20CR2c':
    ic=twcr.load(args.variable,datetime.datetime(args.year,args.month,
                                                args.day,args.hour)+
                               datetime.timedelta(hours=6),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=args.member))
    # Normalise to mean=0, sd=1 (approx)
    if args.variable=='prmsl':
        ic.data -= 101325
        ic.data /= 3000
    elif args.variable=='air.2m':
        ic.data -= 280
        ic.data /= 50
    elif args.variable=='prate':
        pass

else:
    raise ValueError('Source %s is not supported' % args.source)

# Convert to Tensor
ict=tf.convert_to_tensor(ic.data, numpy.float32)

# Write to tfrecord file
sict=tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile,sict)

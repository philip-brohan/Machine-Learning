#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Extract the values at the locations of DWR stations
# Convert them into a TensorFlow tensor.
# Serialise that and store it on $SCRATCH.

import tensorflow as tf
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os

# Get the DWR station locations
import DWR
obs=DWR.load_observations('prmsl',
                          datetime.datetime(1903,10,1,7,30),
                          datetime.datetime(1903,10,1,8,30))

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
    args.opfile=("%s/Machine-Learning-experiments/datasets/DWR/%s/%s/%s/%04d-%02d-%02d:%02d.tfd" %
                       (os.getenv('SCRATCH'),args.source,args.variable,purpose,
                        args.year,args.month,args.day,args.hour))

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

if args.source=='20CR2c':
    ic=twcr.load(args.variable,datetime.datetime(args.year,args.month,
                                                args.day,args.hour),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=args.member))

    #interpolator = iris.analysis.Linear().interpolator(ic, 
    #                               ['latitude', 'longitude'])
    ensemble=[]
    for index, row in obs.iterrows():
        ensemble.append(ic.interpolate(
                        [('latitude',row['latitude']),
                         ('longitude',row['longitude'])],
                        iris.analysis.Linear()).data.item())
    ensemble = numpy.array(ensemble, dtype=numpy.float32)
 
    # Normalise to mean=0, sd=1 (approx)
    if args.variable=='prmsl':
        ensemble -= 101325
        ensemble /= 3000
    elif args.variable=='air.2m':
        ensemble -= 280
        ensemble /= 50
    elif args.variable=='prate':
        pass
        # Don't normalise prate until later
        #ic.data = ic.data+numpy.random.uniform(0,numpy.exp(-11),
        #                                           ic.data.shape)
        #ic.data = numpy.log(ic.data)
        #ic.data += 11.5
        #ic.data = numpy.maximum(ic.data,-7)
        #ic.data = numpy.minimum(ic.data,7)

else:
    raise ValueError('Source %s is not supported' % args.source)

# Convert to Tensor
ict=tf.convert_to_tensor(ensemble, numpy.float32)

# Write to tfrecord file
sict=tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile,sict)

#!/usr/bin/env python

# Store a 20CR2c state in the reduced dimensional space

import tensorflow as tf
tf.enable_eager_execution()
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os
import math
import pickle


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
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=True)
parser.add_argument("--opfile", help="tf data file name",
                    default=None,
                    type=str,required=False)
args = parser.parse_args()
if args.opfile is None:
    args.opfile=(("%s/Machine-Learning-experiments/datasets/latent_space/"+
                  "%04d-%02d-%02d:%02d.tfd") %
                       (os.getenv('SCRATCH'),
                        args.year,args.month,args.day,args.hour))
if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Projection for tensors and plotting
cs=iris.coord_systems.RotatedGeogCS(90,180,0)

# Define cube for regridding data to match the training tensors
def tensor_cube(cbe):
    # Latitudes cover -90 to 90 with 79 values
    lat_values=numpy.arange(-90,91,180/78)
    latitude = iris.coords.DimCoord(lat_values,
                                    standard_name='latitude',
                                    units='degrees_north',
                                    coord_system=cs)
    # Longitudes cover -180 to 180 with 159 values
    lon_values=numpy.arange(-180,181,360/158)
    longitude = iris.coords.DimCoord(lon_values,
                                     standard_name='longitude',
                                     units='degrees_east',
                                     coord_system=cs)
    dummy_data = numpy.zeros((len(lat_values), len(lon_values)))
    dummy_cube = iris.cube.Cube(dummy_data,
                               dim_coords_and_dims=[(latitude, 0),
                                                    (longitude, 1)])
    n_cube=cbe.regrid(dummy_cube,iris.analysis.Linear())
    return(n_cube)




# Normalisation and un-normalisation functions
def normalise_precip(p):
   res=p.copy()
   res.data = res.data*1000+1.001
   res.data = numpy.log(res.data)
   return res
def unnormalise_precip(p):
   res=p.copy()
   res.data = numpy.exp(res.data)
   res.data = (res.data-1.001)/1000
   return res
def normalise_t2m(p):
   res=p.copy()
   res.data -= 280
   res.data /= 50
   return res
def unnormalise_t2m(p):
   res=p.copy()
   res.data *= 50
   res.data += 280
   return res
def normalise_wind(p):
   res=p.copy()
   res.data /= 12
   return res
def unnormalise_wind(p):
   res=p.copy()
   res.data *= 12
   return res
def normalise_prmsl(p):
   res=p.copy()
   res.data -= 101325
   res.data /= 3000
   return res
def unnormalise_prmsl(p):
   res=p.copy()
   res.data *= 3000
   res.data += 101325
   return res

# Load the validation data
prmsl=twcr.load('prmsl',datetime.datetime(args.year,args.month,
                                          args.day,args.hour),
                        version='2c')
prmsl=tensor_cube(prmsl.extract(iris.Constraint(member=1)))
t2m=twcr.load('air.2m',datetime.datetime(args.year,args.month,
                                         args.day,args.hour),
                       version='2c')
t2m=tensor_cube(t2m.extract(iris.Constraint(member=1)))
u10m=twcr.load('uwnd.10m',datetime.datetime(args.year,args.month,
                                            args.day,args.hour),
                          version='2c')
u10m=tensor_cube(u10m.extract(iris.Constraint(member=1)))
v10m=twcr.load('vwnd.10m',datetime.datetime(args.year,args.month,
                                            args.day,args.hour),
                          version='2c')
v10m=tensor_cube(v10m.extract(iris.Constraint(member=1)))

# Convert the validation data into tensor format
t2m_t = tf.convert_to_tensor(normalise_t2m(t2m).data,numpy.float32)
t2m_t = tf.reshape(t2m_t,[79,159,1])
prmsl_t = tf.convert_to_tensor(normalise_prmsl(prmsl).data,numpy.float32)
prmsl_t = tf.reshape(prmsl_t,[79,159,1])
u10m_t = tf.convert_to_tensor(normalise_wind(u10m).data,numpy.float32)
u10m_t = tf.reshape(u10m_t,[79,159,1])
v10m_t = tf.convert_to_tensor(normalise_wind(v10m).data,numpy.float32)
v10m_t = tf.reshape(v10m_t,[79,159,1])

# Get encoded versions of the validation data
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_uk_centred_var/saved_models/"+
                  "Epoch_%04d/encoder") % (
                      os.getenv('SCRATCH'),args.epoch)
encoder=tf.keras.models.load_model(model_save_file,compile=False)
ict = tf.concat([t2m_t,prmsl_t,u10m_t,v10m_t],2) # Now [79,159,4]
ict = tf.reshape(ict,[1,79,159,4])
result = encoder.predict_on_batch(ict)
result = tf.convert_to_tensor(result,numpy.float32)
sict=tf.serialize_tensor(result)
tf.write_file(args.opfile,sict)


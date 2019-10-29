#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Rescale it and rotate the pole.
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

# Rescale makes the dimensions easy to generate with strided convolutions
# Rotation moveds the equator to the boundary - to reduce the problems
#  with boundary conditions.

import tensorflow as tf
tf.enable_eager_execution()
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
    args.opfile=(("%s/Machine-Learning-experiments/datasets/uk_centred+6h/"+
                  "%s/%s/%s/%04d-%02d-%02d:%02d.tfd") %
                       (os.getenv('SCRATCH'),args.source,args.variable,purpose,
                        args.year,args.month,args.day,args.hour))

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Normalisation and un-normalisation functions
def normalise_precip(p):
   res=p.copy()
   res.data = res.data*1000+1.001
   res.data = numpy.log(res.data)
   return res
def normalise_t2m(p):
   res=p.copy()
   res.data -= 280
   res.data /= 50
   return res
def normalise_wind(p):
   res=p.copy()
   res.data /= 12
   return res
def normalise_prmsl(p):
   res=p.copy()
   res.data -= 101325
   res.data /= 3000
   return res

    
# Function to resize and rotate pole
def rr_cube(cbe):
    # Standard pole with UK at centre
    cs=iris.coord_systems.RotatedGeogCS(90.0,180.0,0.0)
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


if args.source=='20CR2c':
    ic=twcr.load(args.variable,datetime.datetime(args.year,args.month,
                                                args.day,args.hour)+
                               datetime.timedelta(hours=6),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=args.member))
    ic=rr_cube(ic)
    # Normalise to range 0-1 (approx)
    if args.variable=='uwnd.10m' or args.variable=='vwnd.10m':
        ic=normalise_wind(ic)
    elif args.variable=='air.2m':
        ic=normalise_t2m(ic)
    elif args.variable=='prate':
        ic=normalise_precip(ic)
    elif args.variable=='prmsl':
        ic=normalise_prmsl(ic)
    
else:
    raise ValueError('Source %s is not supported' % args.source)

# Convert to Tensor
ict=tf.convert_to_tensor(ic.data, numpy.float32)

# Write to tfrecord file
sict=tf.serialize_tensor(ict)
tf.write_file(args.opfile,sict)

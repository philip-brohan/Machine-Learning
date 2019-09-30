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
    args.opfile=(("%s/Machine-Learning-experiments/datasets/uk_centred/"+
                  "%s/%s/%s/%04d-%02d-%02d:%02d.tfd") %
                       (os.getenv('SCRATCH'),args.source,args.variable,purpose,
                        args.year,args.month,args.day,args.hour))

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Normalisation and un-normalisation functions
def normalise_precip(p):
   res=p.copy()
   res.data[res.data<=2.00e-5]=0.79
   res.data[res.data<2.10e-5]=0.81
   res.data[res.data<2.50e-5]=0.83
   res.data[res.data<3.10e-5]=0.85
   res.data[res.data<3.80e-5]=0.87
   res.data[res.data<4.90e-5]=0.89
   res.data[res.data<6.60e-5]=0.91
   res.data[res.data<9.10e-5]=0.93
   res.data[res.data<13.4e-5]=0.95
   res.data[res.data<22.0e-5]=0.97
   res.data[res.data<0.79]=0.99
   return res
def normalise_t2m(p):
   res=p.copy()
   res.data[res.data>300.10]=0.95
   res.data[res.data>299.9]=0.90
   res.data[res.data>298.9]=0.85
   res.data[res.data>297.5]=0.80
   res.data[res.data>295.7]=0.75
   res.data[res.data>293.5]=0.70
   res.data[res.data>290.1]=0.65
   res.data[res.data>287.6]=0.60
   res.data[res.data>283.7]=0.55
   res.data[res.data>280.2]=0.50
   res.data[res.data>277.2]=0.45
   res.data[res.data>274.4]=0.40
   res.data[res.data>272.3]=0.35
   res.data[res.data>268.3]=0.30
   res.data[res.data>261.4]=0.25
   res.data[res.data>254.6]=0.20
   res.data[res.data>249.1]=0.15
   res.data[res.data>244.9]=0.10
   res.data[res.data>240.5]=0.05
   res.data[res.data>0.95]=0.0
   return res
def normalise_wind(p):
   res=p.copy()
   res.data[p.data>=11.76]                  = 0.975
   res.data[(p.data<11.76) & (p.data>=8.48)]= 0.925
   res.data[(p.data<8.48) & (p.data>=6.41)] = 0.875
   res.data[(p.data<6.41) & (p.data>=4.67)] = 0.825
   res.data[(p.data<4.67) & (p.data>=3.42)] = 0.775
   res.data[(p.data<3.42) & (p.data>=2.50)] = 0.725
   res.data[(p.data<2.50) & (p.data>=1.75)] = 0.675
   res.data[(p.data<1.75) & (p.data>=1.10)] = 0.625
   res.data[(p.data<1.10) & (p.data>=0.49)] = 0.575
   res.data[(p.data<0.49) & (p.data>=-0.10)] = 0.525
   res.data[(p.data<-0.10) & (p.data>=-0.69)] = 0.475
   res.data[(p.data<-0.69) & (p.data>=-1.31)] = 0.425
   res.data[(p.data<-1.31) & (p.data>=-1.99)] = 0.375
   res.data[(p.data<-1.99) & (p.data>=-2.75)] = 0.325
   res.data[(p.data<-2.75) & (p.data>=-3.59)] = 0.275
   res.data[(p.data<-3.59) & (p.data>=-4.54)] = 0.225
   res.data[(p.data<-4.54) & (p.data>=-5.69)] = 0.175
   res.data[(p.data<-5.69) & (p.data>=-7.04)] = 0.125
   res.data[(p.data<-7.04) & (p.data>=-8.96)] = 0.075
   res.data[p.data<-8.96]                     = 0.025
   return res

    
# Function to resize and rotate pole
def rr_cube(cbe):
    # Standard pole with UK at centre
    cs=iris.coord_systems.RotatedGeogCS(9.0,180.0,0.0)
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
                               datetime.timedelta(hours=24),
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
    
else:
    raise ValueError('Source %s is not supported' % args.source)

# Convert to Tensor
ict=tf.convert_to_tensor(ic.data, numpy.float32)

# Write to tfrecord file
sict=tf.serialize_tensor(ict)
tf.write_file(args.opfile,sict)

#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Rescale it and move UK to the centre of the field.
# Convert it into a TensorFlow tensor.
# Serialise it and store it on $SCRATCH.

# Rescale makes the dimensions easy to generate with strided convolutions


import tensorflow as tf
tf.enable_eager_execution()
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os

coord_s=iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

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
                       (os.getenv('SCRATCH'),'20CR2c','insolation',purpose,
                        args.year,args.month,args.day,args.hour))

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Normalisation and un-normalisation functions
def normalise_insolation(p):
   res=p.copy()
   res.data /= 25
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


time_constraint=iris.Constraint(time=iris.time.PartialDateTime(
                                year=args.year,
                                month=args.month,
                                day=args.day,
                                hour=args.hour))
ic=iris.load_cube("%s/20CR/version_2c/ensmean/cduvb.1969.nc" % os.getenv('DATADIR'),
                  iris.Constraint(name='3-hourly Clear Sky UV-B Downward Solar Flux') &
                  time_constraint)
ic.coord('latitude').coord_system=coord_s
ic.coord('longitude').coord_system=coord_s
ic=rr_cube(ic)
ic=normalise_insolation(ic)    

# Convert to Tensor
ict=tf.convert_to_tensor(ic.data, numpy.float32)

# Write to tfrecord file
sict=tf.serialize_tensor(ict)
tf.write_file(args.opfile,sict)

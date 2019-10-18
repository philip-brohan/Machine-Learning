#!/usr/bin/env python

# Read in a field from 20CR as an Iris cube.
# Extract the values at the locations of DWR stations
# Convert them into a TensorFlow tensor.
# Serialise that and store it on $SCRATCH.

import numpy
import pandas

import iris
import datetime
import argparse
import os
import pickle

# Get the DWR station locations
import DWR
obs=DWR.load_observations('prmsl',
                          datetime.datetime(1903,10,1,7,30),
                          datetime.datetime(1903,10,1,8,30))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year",
                    type=int,required=True)
parser.add_argument("--member", help="Ensemble member",
                    default=1,type=int,required=False)
parser.add_argument("--variable", help="variable name",
                    default='prmsl',type=str,required=False)
parser.add_argument("--opfile", help="pkl data file name",
                    default=None,
                    type=str,required=False)
args = parser.parse_args()
if args.opfile is None:
    args.opfile=("%s/Machine-Learning-experiments/datasets/DWR/20CRv2c/%s/%04d.pkl" %
                       (os.getenv('SCRATCH'),args.variable,args.year))

if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

ic=iris.load_cube('/scratch/hadpb/20CR/version_2c/%04d/%s.nc' % (args.year,args.variable),
                  iris.Constraint(coord_values={
                    'ensemble member number':lambda cell: cell==1}))
tc=ic.coords('time')[0]
df=pandas.DataFrame({'Date':tc.units.num2date(tc.points)})

for index, row in obs.iterrows():
    df[row['name']]=ic.interpolate(
                    [('latitude',row['latitude']),
                     ('longitude',row['longitude'])],
                     iris.analysis.Linear()).data.data

pickle.dump( df, open( args.opfile, "wb" ) )

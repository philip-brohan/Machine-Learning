#!/usr/bin/env python

# Check the annual and diurnal cycle inference model

import os
import pickle
import datetime
import numpy
import tensorflow as tf
tf.enable_eager_execution()
import IRData.twcr as twcr
import iris
import math

import matplotlib
from matplotlib.backends.backend_agg import \
                 FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="Start time: YYYY-MM-DD:HH",
                    type=str,default='2006-01-01:00',required=False)
parser.add_argument("--end", help="End time: YYYY-MM-DD:HH",
                    type=str,default='2006-12-31:18',required=False)
parser.add_argument("--epoch", help="Model epoch to check",
                    type=int,required=False,default=10)
args = parser.parse_args()
args.opdir=(("%s/Machine-Learning-experiments/"+
             "convolutional_autoencoder_perturbations/"+
             "/check_cycles/saved_models/Epoch_%04d") %
               (os.getenv('SCRATCH'),args.epoch))
args.start=datetime.datetime.strptime(args.start,"%Y-%m-%d:%H")
args.end  =datetime.datetime.strptime(args.end,  "%Y-%m-%d:%H")

# Load the saved model
save_file="%s/encoder" % args.opdir
encoder=tf.keras.models.load_model(save_file)

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

# Normalisation functions
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

#  load the 20CR data and calculate the cycles
def get_cycles(dte):
            
    ic=twcr.load('air.2m',datetime.datetime(dte.year,dte.month,
                                            dte.day,dte.hour),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=1))
    ic=rr_cube(ic)
    ic=normalise_t2m(ic)
    t2m=tf.convert_to_tensor(ic.data, numpy.float32)
    t2m = tf.reshape(t2m,[79,159,1])
    ic=twcr.load('prmsl',datetime.datetime(dte.year,dte.month,
                                            dte.day,dte.hour),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=1))
    ic=rr_cube(ic)
    ic=normalise_prmsl(ic)
    prmsl=tf.convert_to_tensor(ic.data, numpy.float32)
    prmsl = tf.reshape(prmsl,[79,159,1])
    ic=twcr.load('uwnd.10m',datetime.datetime(dte.year,dte.month,
                                            dte.day,dte.hour),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=1))
    ic=rr_cube(ic)
    ic=normalise_wind(ic)
    uwnd=tf.convert_to_tensor(ic.data, numpy.float32)
    uwnd = tf.reshape(uwnd,[79,159,1])
    ic=twcr.load('vwnd.10m',datetime.datetime(dte.year,dte.month,
                                            dte.day,dte.hour),
                 version='2c')
    ic=ic.extract(iris.Constraint(member=1))
    ic=rr_cube(ic)
    ic=normalise_wind(ic)
    vwnd=tf.convert_to_tensor(ic.data, numpy.float32)
    vwnd = tf.reshape(vwnd,[79,159,1])
    ict = tf.concat([t2m,prmsl,uwnd,vwnd],2)
    ict = tf.reshape(ict,[1,79,159,4])

    res=encoder.predict_on_batch(ict)
    return(res)

# For each time in the validation period,
a_annual  = []
a_diurnal = []
e_annual  = []
e_diurnal = []
dates     = []
current=args.start
mdays=[0,31,59,90,120,151,181,212,243,273,304,334]
while current<args.end:
    dates.append(current)
    a_diurnal.append(current.hour/24)
    dy=current.day
    if current.month==2 and current.day==29: dy=28
    a_annual.append(math.sin((3.141592*(mdays[current.month-1]+dy)/365)))
    encoded = get_cycles(current)
    e_annual.append(encoded[0,0])
    e_diurnal.append(encoded[0,1])
    current += datetime.timedelta(hours=246)

# Make the plot
aspect=16.0/9.0
fig=Figure(figsize=(10.8*aspect,10.8),  # Width, Height (inches)
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)
font = {'family' : 'sans-serif',
        'sans-serif' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

# Bottom axes - annual v. time
ax=fig.add_axes([0.05,0.05,0.945,0.4])
ax.set_xlim(min(dates),
            max(dates))
ax.set_ylim(-0.1,1.1)
ax.set_ylabel('Annual Cycle')
ax.grid()
ax.plot(dates,a_annual,
                markersize=5,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax.plot(dates,e_annual,
                markersize=5,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
# Next axes - t2m v. time
ax2=fig.add_axes([0.05,0.5,0.945,0.4])
ax2.set_xlim(min(dates),
            max(dates))
ax2.set_ylim(-0.1,0.8)
ax2.set_ylabel('Diurnal Cycle')
ax2.grid()
ax2.plot(dates,a_diurnal,
                markersize=5,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax2.plot(dates,e_diurnal,
                markersize=5,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)

fig.savefig('Encodings.png')

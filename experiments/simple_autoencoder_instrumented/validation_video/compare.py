#!/usr/bin/env python

# Overplot original and encoded pressure fields.
# Can be run at any epoch and any date - for video diagnostics.

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

import Meteorographica as mg

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Model at which epoch?",
                    type=int,required=True)
parser.add_argument("--year", help="Comparison year",
                    type=int,required=True)
parser.add_argument("--month", help="Comparison month",
                    type=int,required=True)
parser.add_argument("--day", help="Comparison day",
                    type=int,required=True)
parser.add_argument("--hour", help="Comparison hour",
                    type=int,required=True)
args = parser.parse_args()

# Get the 20CR data
ic=twcr.load('prmsl',datetime.datetime(args.year,
                                       args.month,
                                       args.day,
                                       args.hour),
                           version='2c')
ic=ic.extract(iris.Constraint(member=1))

# Get the autoencoder at the chosen epoch
model_save_file = ("%s/Machine-Learning-experiments/"+
                   "simple_autoencoder_instrumented/"+
                   "saved_models/Epoch_%04d") % (
                         os.getenv('SCRATCH'),args.epoch)
autoencoder=tf.keras.models.load_model(model_save_file)

# Normalisation - Pa to mean=0, sd=1 - and back
def normalise(x):
   x -= 101325
   x /= 3000
   return x

def unnormalise(x):
   x *= 3000
   x += 101325
   return x

fig=Figure(figsize=(19.2,10.8),  # 1920x1080, HD
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Map showing original and reconstructed fields
projection=ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=90.0)
ax_map=fig.add_axes([0,0,1,1],projection=projection)
ax_map.set_axis_off()
extent=[-180,180,-90,90]
ax_map.set_extent(extent, crs=projection)
matplotlib.rc('image',aspect='auto')

# Run the data through the autoencoder and convert back to iris cube
pm=ic.copy()
pm.data=normalise(pm.data)
ict=tf.convert_to_tensor(pm.data, numpy.float32)
ict=tf.reshape(ict,[1,91*180]) # ????
result=autoencoder.predict_on_batch(ict)
result=tf.reshape(result,[91,180])
pm.data=unnormalise(result)

# Background, grid and land
ax_map.background_patch.set_facecolor((0.88,0.88,0.88,1))
#mg.background.add_grid(ax_map)
land_img_orig=ax_map.background_img(name='GreyT', resolution='low')

# original pressures as red contours
mg.pressure.plot(ax_map,ic,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,7),
                 colors='red',
                 label=False,
                 linewidths=2)
# Encoded pressures as blue contours
mg.pressure.plot(ax_map,pm,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,7),
                 colors='blue',
                 label=False,
                 linewidths=2)

mg.utils.plot_label(ax_map,
                    '%04d-%02d-%02d:%02d' % (args.year,
                                             args.month,
                                             args.day,
                                             args.hour),
                    facecolor=(0.88,0.88,0.88,0.9),
                    fontsize=16,
                    x_fraction=0.98,
                    y_fraction=0.03,
                    verticalalignment='bottom',
                    horizontalalignment='right')

# Render the figure as a png
figfile=("%s/Machine-Learning-experiments/"+
                   "simple_autoencoder_instrumented/"+
                   "images/%02d%02d%02d%02d_%04d.png") % (
                          os.getenv('SCRATCH'),
                          args.year,args.month,
                          args.day,args.hour,
                          args.epoch)
if not os.path.isdir(os.path.dirname(figfile)):
    os.makedirs(os.path.dirname(figfile))
fig.savefig(figfile)


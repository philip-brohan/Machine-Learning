#!/usr/bin/env python

# Compare a 20CRv2c prmsl field with the same field passed through
#   the autoencoder.

import tensorflow as tf
tf.enable_eager_execution()
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import os

import Meteorographica as mg

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs

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
parser.add_argument("--epoch", help="Model at which epoch?",
                    type=int,required=True)

args = parser.parse_args()

# Get the 20CR data
ic=twcr.load(args.variable,datetime.datetime(args.year,args.month,
                                            args.day,args.hour),
                           version=args.version)
ic=ic.extract(iris.Constraint(member=args.member))

# Get the autoencoder
model_save_file=("%s/Machine-Learning-experiments/simple_autoencoder/"+
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

# Run the data through the autoencoder and convert back to iris cube
pm=ic.copy()
pm.data=normalise(pm.data)
ict=tf.convert_to_tensor(pm.data, numpy.float32)
ict=tf.reshape(ict,[1,91*180]) # ????
result=autoencoder.predict_on_batch(ict)
result=tf.reshape(result,[91,180])
pm.data=unnormalise(result)

# Make a comparison plot - original on top, encoded below
fig=Figure(figsize=(15,15*1.06/1.04),  # Width, Height (inches)
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Global projection
projection=ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=90.0)
extent=[-180,180,-90,90]

# Top half for the originals
ax_orig=fig.add_axes([0.02,0.51,0.96,0.47],projection=projection)
ax_orig.set_axis_off()
ax_orig.set_extent(extent, crs=projection)
ax_post=fig.add_axes([0.02,0.02,0.96,0.47],projection=projection)
ax_post.set_axis_off()
ax_post.set_extent(extent, crs=projection)

# Background, grid and land for both
ax_orig.background_patch.set_facecolor((0.88,0.88,0.88,1))
ax_post.background_patch.set_facecolor((0.88,0.88,0.88,1))
mg.background.add_grid(ax_orig)
mg.background.add_grid(ax_post)
land_img_orig=ax_orig.background_img(name='GreyT', resolution='low')
land_img_post=ax_post.background_img(name='GreyT', resolution='low')

# Plot the pressures as contours
mg.pressure.plot(ax_orig,ic,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,7),
                 colors='blue',
                 label=True,
                 linewidths=2)
mg.pressure.plot(ax_post,pm,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,7),
                 colors='blue',
                 label=True,
                 linewidths=2)

# Mark the data used
mg.utils.plot_label(ax_post,
              '%04d-%02d-%02d:%02d' % (args.year,args.month,args.day,args.hour),
              facecolor=fig.get_facecolor(),
              x_fraction=0.98,
              horizontalalignment='right')


# Render the figure as a png
fig.savefig("comparison_%04d-%02d-%02d:%02d.png" % 
             (args.year,args.month,args.day,args.hour))

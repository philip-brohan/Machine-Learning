#!/usr/bin/env python

# Compare a 20CRv2c prmsl field with the same field passed through
#   the autoencoder.

import tensorflow as tf
tf.enable_eager_execution()
import numpy

import ML_Utilities

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

args = parser.parse_args()

# Get the 20CR data
ic=twcr.load('prmsl',datetime.datetime(args.year,args.month,
                                            args.day,args.hour),
                           version='2c')
ic=ic.extract(iris.Constraint(member=args.member))

# Normalisation - Pa to mean=0, sd=1 - and back
normalise=ML_Utilities.get_normalise_function(source='20CR2c',
                                              variable='prmsl')
unnormalise=ML_Utilities.get_unnormalise_function(source='20CR2c',
                                                  variable='prmsl')

# Get the original autoencoder
model_save_file=("%s/Machine-Learning-experiments/simple_autoencoder/"+
                 "saved_models/Epoch_%04d") % (
                    os.getenv('SCRATCH'),100)
autoencoder_original=tf.keras.models.load_model(model_save_file)

# Run the data through the original autoencoder and convert back to iris cube
pm_original=ic.copy()
pm_original.data=normalise(pm_original.data)
ict=tf.convert_to_tensor(pm_original.data, numpy.float32)
ict=tf.reshape(ict,[1,91*180]) # ????
result=autoencoder_original.predict_on_batch(ict)
result=tf.reshape(result,[91,180])
pm_original.data=unnormalise(result)

# Same with the modified autoencoder
model_save_file=(("%s/Machine-Learning-experiments/"+
                  "simple_autoencoder_perturbations/regularized/"+
                 "saved_models/Epoch_%04d")) % (
                    os.getenv('SCRATCH'),100)
autoencoder_modified=tf.keras.models.load_model(model_save_file)

# Run the data through the autoencoder and convert back to iris cube
pm_modified=ic.copy()
pm_modified.data=normalise(pm_modified.data)
ict=tf.convert_to_tensor(pm_modified.data, numpy.float32)
ict=tf.reshape(ict,[1,91*180]) # ????
result=autoencoder_modified.predict_on_batch(ict)
result=tf.reshape(result,[91,180])
pm_modified.data=unnormalise(result)

# Make a comparison plot - original, then old autoencoder, then new one
fig=Figure(figsize=(15,15*1.06/1.04*1.5),  # Width, Height (inches)
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

# Top third for the originals
ax_orig=fig.add_axes([0.02,0.67,0.96,0.32],projection=projection)
ax_orig.set_axis_off()
ax_orig.set_extent(extent, crs=projection)
ax_old=fig.add_axes([0.02,0.34,0.96,0.32],projection=projection)
ax_old.set_axis_off()
ax_old.set_extent(extent, crs=projection)
ax_new=fig.add_axes([0.02,0.01,0.96,0.32],projection=projection)
ax_new.set_axis_off()
ax_new.set_extent(extent, crs=projection)

# Background, grid and land
ax_orig.background_patch.set_facecolor((0.88,0.88,0.88,1))
ax_old.background_patch.set_facecolor((0.88,0.88,0.88,1))
ax_new.background_patch.set_facecolor((0.88,0.88,0.88,1))
mg.background.add_grid(ax_orig)
mg.background.add_grid(ax_old)
mg.background.add_grid(ax_new)
land_img_orig=ax_orig.background_img(name='GreyT', resolution='low')
land_img_old=ax_old.background_img(name='GreyT', resolution='low')
land_img_new=ax_new.background_img(name='GreyT', resolution='low')

# Plot the pressures as contours
mg.pressure.plot(ax_orig,ic,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,10),
                 colors='blue',
                 label=True,
                 linewidths=2)
mg.pressure.plot(ax_old,pm_original,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(1010,1040,2),
                 colors='blue',
                 label=True,
                 linewidths=2)
mg.pressure.plot(ax_new,pm_modified,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(1010,1040,2),
                 colors='blue',
                 label=True,
                 linewidths=2)

# Mark the data used
mg.utils.plot_label(ax_new,
              '%04d-%02d-%02d:%02d' % (args.year,args.month,args.day,args.hour),
              facecolor=fig.get_facecolor(),
              x_fraction=0.98,
              horizontalalignment='right')


# Render the figure as a png
fig.savefig("comparison_%04d-%02d-%02d:%02d.png" % 
             (args.year,args.month,args.day,args.hour))

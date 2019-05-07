#!/usr/bin/env python

# Show the autoencoder weights.

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

# Get a 20CR data for the grid metadata
ic=twcr.load('prmsl',datetime.datetime(1969,3,12,6),
                           version='2c')
ic=ic.extract(iris.Constraint(member=1))

# Get the single_neuron autoencoder
model_save_file=(("%s/Machine-Learning-experiments/"+
                  "simple_autoencoder_introspection/single_neuron/"+
                 "saved_models/Epoch_%04d")) % (
                     os.getenv('SCRATCH'),500)
autoencoder=tf.keras.models.load_model(model_save_file)

# Make a comparison plot - Input, hidden, and output weights
fig=Figure(figsize=(10,13),  # Width, Height (inches)
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

# Top half for the input weights
ax_input=fig.add_axes([0.02,0.55,0.96,0.43],projection=projection)
ax_input.set_axis_off()
ax_input.set_extent(extent, crs=projection)
ax_input.background_patch.set_facecolor((0.88,0.88,0.88,1))

# Input weights as iris cube
w_in=ic.copy()
w_in.data=autoencoder.get_weights()[0].reshape(ic.data.shape)
# Plot the weights as a colour map

lats = w_in.coord('latitude').points
lons = w_in.coord('longitude').points-180
prate_img=ax_input.pcolorfast(lons, lats, w_in.data, 
                        cmap='coolwarm',
                        vmin=numpy.mean(w_in.data)-numpy.std(w_in.data)*3,
                        vmax=numpy.mean(w_in.data)+numpy.std(w_in.data)*3,
                        )

# Give the mean and sd
mg.utils.plot_label(ax_input,
              'Mean = %5.2f\nsd = %5.2f' % (numpy.mean(w_in.data),
                                          numpy.std(w_in.data)),
              facecolor=fig.get_facecolor(),
              x_fraction=0.98,
              y_fraction=0.04,
              horizontalalignment='right',
              verticalalignment='bottom')

# Same for output weights at bottom
ax_output=fig.add_axes([0.02,0.02,0.96,0.43],projection=projection)
ax_output.set_axis_off()
ax_output.set_extent(extent, crs=projection)
ax_output.background_patch.set_facecolor((0.88,0.88,0.88,1))

# Output weights as iris cube
w_out=ic.copy()
w_out.data=autoencoder.get_weights()[2].reshape(ic.data.shape)
# Plot the weights as a colour map

lats = w_out.coord('latitude').points
lons = w_out.coord('longitude').points-180
prate_img=ax_output.pcolorfast(lons, lats, w_out.data, 
                        cmap='coolwarm',
                        vmin=numpy.mean(w_out.data)-numpy.std(w_out.data)*3,
                        vmax=numpy.mean(w_out.data)+numpy.std(w_out.data)*3,
                        )

# Give the mean and sd
mg.utils.plot_label(ax_output,
              'Mean = %5.2f\nsd = %5.2f' % (numpy.mean(w_out.data),
                                            numpy.std(w_out.data)),
              facecolor=fig.get_facecolor(),
              x_fraction=0.98,
              y_fraction=0.04,
              horizontalalignment='right',
              verticalalignment='bottom')

# Render the figure as a png
fig.savefig("weights.png")

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
import math

import Meteorographica as mg

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs

# Where on the plot to put each axes
def axes_geom(layer=0,channel=0,nchannels=36):
     if layer==0: 
         base=[0.0,0.6,1.0,0.4]
     else:
         base=[0.0,0.0,1.0,0.4]
     ncol=math.sqrt(nchannels)
     nr=channel//ncol
     nc=channel-ncol*nr
     nr=ncol-1-nr # Top down
     geom=[base[0]+(base[2]/ncol)*0.95*nc,
           base[1]+(base[3]/ncol)*0.95*nr,
           (base[2]/ncol)*0.95,
           (base[3]/ncol)*0.95]
     geom[0] += (0.05*base[2]/(ncol+1))*(nc+1)
     geom[1] += (0.05*base[3]/(ncol+1))*(nr+1)
     return geom

# Plot a single set of weights
def plot_weights(weights,layer=0,channel=0,nchannels=36,
                 vmin=None,vmax=None):
    ax_input=fig.add_axes(axes_geom(layer=layer,
                                    channel=channel,
                                    nchannels=nchannels),
                          projection=projection)
    ax_input.set_axis_off()
    ax_input.set_extent(extent, crs=projection)
    ax_input.background_patch.set_facecolor((0.88,0.88,0.88,1))

    lats = w_in.coord('latitude').points
    lons = w_in.coord('longitude').points-180
    prate_img=ax_input.pcolorfast(lons, lats, w_in.data, 
                            cmap='coolwarm',
                            vmin=vmin,
                            vmax=vmax,
                            )

# Plot the hidden layer weights
def plot_hidden(weights):
     # Single axes - var v. time
     ax=fig.add_axes([0.05,0.425,0.9,0.15])
     # Axes ranges from data
     ax.set_xlim(-0.6,len(weights)-0.4)
     ax.set_ylim(0,numpy.max(numpy.abs(weights))*1.05)
     ax.bar(x=range(len(weights)),
            height=numpy.abs(weights[order]),
            color='grey',
            tick_label=order)

# Get a 20CR data for the grid metadata
ic=twcr.load('prmsl',datetime.datetime(1969,3,12,6),
                           version='2c')
ic=ic.extract(iris.Constraint(member=1))

# Get the 9 neuron autoencoder
model_save_file=(("%s/Machine-Learning-experiments/"+
                  "simple_autoencoder_activations/relu/"+
                 "saved_models/Epoch_%04d")) % (
                     os.getenv('SCRATCH'),100)
autoencoder=tf.keras.models.load_model(model_save_file)

# Get the order of the hidden weights - most to least important
order=numpy.argsort(numpy.abs(autoencoder.get_weights()[1]))[::-1]

# Make a comparison plot - Input, hidden, and output weights
fig=Figure(figsize=(10,12),  # Width, Height (inches)
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Hidden layer
plot_hidden(autoencoder.get_weights()[1])

# Global projection
projection=ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=90.0)
extent=[-180,180,-90,90]

for layer in [0,2]:
    w_l=autoencoder.get_weights()[layer]
    vmin=numpy.mean(w_l)-numpy.std(w_l)*3
    vmax=numpy.mean(w_l)+numpy.std(w_l)*3
    count=0
    for channel in order:
        w_in=ic.copy()
        if layer==0:
            w_in.data=w_l[:,channel].reshape(ic.data.shape)
        else:
            w_in.data=w_l[channel,:].reshape(ic.data.shape)
        w_in.data *= numpy.sign(autoencoder.get_weights()[1][channel])
        plot_weights(w_in,layer=layer,channel=count,nchannels=36,
                     vmin=vmin,vmax=vmax)
        count += 1
        
# Render the figure as a png
fig.savefig("weights.png")

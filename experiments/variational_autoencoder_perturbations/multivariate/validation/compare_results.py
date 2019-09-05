#!/usr/bin/env python

# Model training results plot

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
from pandas import qcut

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=True)
args = parser.parse_args()

# Function to resize and rotate pole
def rr_cube(cbe):
    # Use the Cassini projection (boundary is the equator)
    cs=iris.coord_systems.RotatedGeogCS(0.0,60.0,270.0)
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

# Function to do the multivariate plot
def three_plot(ax,prmsl,prate,t2m):
    ax.set_axis_off() # Don't want surrounding x and y axis
    ax.background_patch.set_facecolor((0.88,0.88,0.88,1))
    land_img=ax.background_img(name='GreyT', resolution='low')
    # Plot the temperature
    plot_cube=mg.utils.dummy_cube(ax,0.25)
    t2m = t2m.regrid(plot_cube,iris.analysis.Linear())
    # Re-map to highlight small differences
    s=t2m.data.shape
    t2m.data=qcut(t2m.data.flatten(),20,labels=False).reshape(s)
    # Plot as a colour map
    lats = t2m.coord('latitude').points
    lons = t2m.coord('longitude').points
    t2m_img=ax.pcolorfast(lons, lats, t2m.data,
                          cmap='coolwarm',
                          vmin=0,
                          vmax=20,
                          alpha=0.5)
    # Also pressure
    mg.pressure.plot(ax,prmsl,scale=0.01,resolution=0.25,
                     linewidths=1)
    # Also precip
    mg.precipitation.plot(ax,prate,resolution=0.25,vmin=-0.01,vmax=0.04)

 
# Get the 20CR data
prmsl=twcr.load('prmsl',datetime.datetime(2009,3,12,18),
                           version='2c')
prmsl=rr_cube(prmsl.extract(iris.Constraint(member=1)))
prate=twcr.load('prate',datetime.datetime(2009,3,12,18),
                           version='2c')
prate=rr_cube(prate.extract(iris.Constraint(member=1)))
rh   =twcr.load('rh9950',datetime.datetime(2009,3,12,18),
                           version='2c')
rh   =rr_cube(rh.extract(iris.Constraint(member=1)))
t2m=twcr.load('air.2m',datetime.datetime(2009,3,12,18),
                           version='2c')
t2m=rr_cube(t2m.extract(iris.Constraint(member=1)))

# Get the autoencoder
model_save_file=("%s/Machine-Learning-experiments/"+
                  "variational_autoencoder_perturbations/"+
                  "multivariate/saved_models/Epoch_%04d/autoencoder") % (
                      os.getenv('SCRATCH'),args.epoch)
autoencoder=tf.keras.models.load_model(model_save_file,compile=False)

# Get the encoded fields 
prmsl_t = prmsl.copy()
prmsl_t.data -= 101325
prmsl_t.data /= 3000
prmsl_t = tf.convert_to_tensor(prmsl_t.data, numpy.float32)
prmsl_t = tf.reshape(prmsl_t,[79,159,1])
t2m_t = t2m.copy()
t2m_t.data -= 280
t2m_t.data /=  50
t2m_t = tf.convert_to_tensor(t2m_t.data, numpy.float32)
t2m_t = tf.reshape(t2m_t,[79,159,1])
prate_t = prate.copy()
s=prate_t.shape
prate_t.data=(prate_t.data*1000+rh.data/100+
                 numpy.random.normal(0,0.05,len(rh.data.flatten())).reshape(s))
prate_t.data -= 1
prate_t = tf.convert_to_tensor(prate_t.data, numpy.float32)
prate_t = tf.reshape(prate_t,[79,159,1])
ict = tf.concat([prmsl_t,t2m_t,prate_t],2) # Now [79,159,3]
ict = tf.reshape(ict,[1,79,159,3])
result = autoencoder.predict_on_batch(ict)
result = tf.reshape(result,[79,159,3])
prmsl_r = prmsl.copy()
prmsl_r.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
prmsl_u=prmsl_r.copy()
prmsl_r.data *= 3000
prmsl_r.data += 101325
t2m_r = t2m.copy()
t2m_r.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
t2m_u=t2m_r.copy()
t2m_r.data *= 50
t2m_r.data += 280
prate_r = prate.copy()
prate_r.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
prate_u=prate_r.copy()
prate_r.data /= 1000
prate_r.data[prate_r.data<0]=0

fig=Figure(figsize=(9.6*1.2,10.8),
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Two maps, original and reconstructed
matplotlib.rc('image',aspect='auto')
projection=ccrs.RotatedPole(pole_longitude=60.0,
                            pole_latitude=0.0,
                            central_rotated_longitude=270.0)
extent=[-180,180,-90,90]
ax_original=fig.add_axes([0.005,0.525,0.75,0.45],projection=projection)
ax_original.set_extent(extent, crs=projection)
three_plot(ax_original,prmsl,prate,t2m)
ax_reconstructed=fig.add_axes([0.005,0.025,0.75,0.45],projection=projection)
ax_reconstructed.set_extent(extent, crs=projection)
three_plot(ax_reconstructed,prmsl_r,prate_r,t2m_r)

# Scatterplot of encoded v original
def plot_scatter(ax,ic,pm):
    dmin=min(ic.min(),pm.min())
    dmax=max(ic.max(),pm.max())
    dmean=(dmin+dmax)/2
    dmax=dmean+(dmax-dmean)*1.05
    dmin=dmean-(dmean-dmin)*1.05
    ax.set_xlim(dmin,dmax)
    ax.set_ylim(dmin,dmax)
    ax.scatter(x=pm.flatten(),
               y=ic.flatten(),
               c='black',
               alpha=0.25,
               marker='.',
               s=2)
    ax.set(ylabel='Original', 
           xlabel='Encoded')
    ax.grid(color='black',
            alpha=0.2,
            linestyle='-', 
            linewidth=0.5)
    
ax_prmsl=fig.add_axes([0.82,0.82,0.16,0.17])
plot_scatter(ax_prmsl,prmsl_t.numpy(),prmsl_u.data) # Normalised units
ax_t2m=fig.add_axes([0.82,0.575,0.16,0.17])
plot_scatter(ax_t2m,t2m_t.numpy(),t2m_u.data)
ax_prate=fig.add_axes([0.82,0.33,0.16,0.17])
plot_scatter(ax_prate,prate_t.numpy(),prate_u.data)

# Plot the training history
history_save_file=("%s/Machine-Learning-experiments/"+
                   "variational_autoencoder_perturbations/"+
                   "multivariate/saved_models/history_to_%04d.pkl") % (
                       os.getenv('SCRATCH'),args.epoch)
history=pickle.load( open( history_save_file, "rb" ) )
ax=fig.add_axes([0.82,0.05,0.16,0.21])
# Axes ranges from data
ax.set_xlim(0,len(history['loss']))
ax.set_ylim(0,numpy.max(numpy.concatenate((history['loss'],
                                           history['val_loss']))))
ax.set(xlabel='Epoch', 
       ylabel='Loss (grey) and validation loss (black)')
ax.grid(color='black',
        alpha=0.2,
        linestyle='-', 
        linewidth=0.5)
ax.plot(range(len(history['loss'])),
        history['loss'],
        color='grey',
        linestyle='-',
        linewidth=2)
ax.plot(range(len(history['val_loss'])),
        history['val_loss'],
        color='black',
        linestyle='-',
        linewidth=2)


# Render the figure as a png
fig.savefig("comparison_results.png")


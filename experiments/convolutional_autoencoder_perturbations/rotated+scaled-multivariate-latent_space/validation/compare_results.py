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

# Function to do a contour comparison
def contour_plot(ax,original,reconstructed,scale,levels):
    ax.set_axis_off() # Don't want surrounding x and y axis
    ax.background_patch.set_facecolor((0.88,0.88,0.88,1))
    #land_img=ax.background_img(name='GreyT', resolution='low')
    # Original
    mg.pressure.plot(ax,original,scale=scale,resolution=0.25,
                     levels=levels,
                     colors='red',
                     label=False,
                     linewidths=1)
    # Reconstructed
    mg.pressure.plot(ax,reconstructed,scale=scale,resolution=0.25,
                     levels=levels,
                     colors='blue',
                     label=False,
                     linewidths=1)

 
# Get the 20CR data
prmsl=twcr.load('prmsl',datetime.datetime(2010,3,12,18),
                           version='2c')
prmsl=rr_cube(prmsl.extract(iris.Constraint(member=1)))
z500=twcr.load('z500',datetime.datetime(2010,3,12,18),
                           version='2c')
z500=rr_cube(z500.extract(iris.Constraint(member=1)))
t2m=twcr.load('air.2m',datetime.datetime(2010,3,12,18),
                           version='2c')
t2m=rr_cube(t2m.extract(iris.Constraint(member=1)))

# Get the autoencoder
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_latent/saved_models/"+
                  "Epoch_%04d/autoencoder") % (
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
z500_t = z500.copy()
z500_t.data -= 5300
z500_t.data /= 600
z500_t = tf.convert_to_tensor(z500_t.data, numpy.float32)
z500_t = tf.reshape(z500_t,[79,159,1])
ict = tf.concat([prmsl_t,t2m_t,z500_t],2) # Now [79,159,3]
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
z500_r = z500.copy()
z500_r.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
z500_u=z500_r.copy()
z500_r.data *= 600
z500_r.data += 5300

fig=Figure(figsize=(19.2,10.8),
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Pressure map
matplotlib.rc('image',aspect='auto')
projection=ccrs.RotatedPole(pole_longitude=60.0,
                            pole_latitude=0.0,
                            central_rotated_longitude=270.0)
extent=[-180,180,-90,90]
ax_prmsl=fig.add_axes([0.01,0.505,0.485,0.485],projection=projection)
ax_prmsl.set_extent(extent, crs=projection)
contour_plot(ax_prmsl,prmsl,prmsl_r,scale=0.01,
                levels=numpy.arange(870,1050,7))
ax_t2m=fig.add_axes([0.505,0.51,0.485,0.485],projection=projection)
ax_t2m.set_extent(extent, crs=projection)
contour_plot(ax_t2m,t2m,t2m_r,scale=1.0,
                levels=numpy.arange(230,310,10))
ax_z500=fig.add_axes([0.01,0.01,0.485,0.485],projection=projection)
ax_z500.set_extent(extent, crs=projection)
contour_plot(ax_z500,z500,z500_r,scale=1.0,
                levels=numpy.arange(4700,6500,200))

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
    
ax_prmsl_s=fig.add_axes([0.55,0.3,0.15,0.2])
plot_scatter(ax_prmsl_s,prmsl_t.numpy(),prmsl_u.data) # Normalised units
ax_t2m_s=fig.add_axes([0.79,0.3,0.15,0.2])
plot_scatter(ax_t2m_s,t2m_t.numpy(),t2m_u.data)
ax_z500_s=fig.add_axes([0.55,0.04,0.15,0.2])
plot_scatter(ax_z500_s,z500_t.numpy(),z500_u.data)

# Plot the training history
history_save_file=("%s/Machine-Learning-experiments/"+
                   "convolutional_autoencoder_perturbations/"+
                   "multivariate_latent/saved_models/"+
                   "history_to_%04d.pkl") % (
                       os.getenv('SCRATCH'),args.epoch)
history=pickle.load( open( history_save_file, "rb" ) )
ax=fig.add_axes([0.79,0.04,0.15,0.2])
# Axes ranges from data
for idx in range(len(history['loss'])):
    history['loss'][idx]=numpy.mean(history['loss'][idx])
    history['val_loss'][idx]=numpy.mean(history['loss'][idx])
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


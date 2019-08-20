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

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs

# Normalisation - Pa to mean=0, sd=1 - and back
def normalise(x):
   x -= 101325
   x /= 3000
   return x

def unnormalise(x):
   x *= 3000
   x += 101325
   return x

class ResizeLayer(tf.keras.layers.Layer):
   def __init__(self, newsize=None, **kwargs):
      super(ResizeLayer, self).__init__(**kwargs)
      self.resize_newsize = newsize
   def call(self, input):
      return tf.image.resize_images(input, self.resize_newsize,
                                    align_corners=True)
   def get_config(self):
      return {'newsize': self.resize_newsize}

year=1916
month=3
day=12
hour=6

# Get the 20CR data
#ic=twcr.load('prmsl',datetime.datetime(2009,3,12,18),
ic=twcr.load('prmsl',datetime.datetime(year,month,day,hour),
                           version='2c')
ic=ic.extract(iris.Constraint(member=1))
#ic=ic.collapsed('member', iris.analysis.MEAN)

# Make the fake observations
obs=twcr.load_observations_fortime(datetime.datetime(1916,3,12,6),
                                   version='2c')
ensemble=[]
for index, row in obs.iterrows():
    ensemble.append(ic.interpolate(
                    [('latitude',row['Latitude']),
                     ('longitude',row['Longitude'])],
                    iris.analysis.Linear()).data.item())
ensemble = numpy.array(ensemble, dtype=numpy.float32)
ensemble = normalise(ensemble)
obs_t = tf.convert_to_tensor(ensemble, numpy.float32)
obs_t = tf.reshape(obs_t,[1,488])

# Get the assimilation model
model_save_file=("%s/Machine-Learning-experiments/"+
                 "oldstyle_assimilation_1916/"+
                 "saved_models/Epoch_%04d") % (
                    os.getenv('SCRATCH'),50)
autoencoder=tf.keras.models.load_model(model_save_file,
                                       custom_objects={'ResizeLayer': ResizeLayer})

fig=Figure(figsize=(9.6,10.8),  # 1/2HD
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Top - map showing original and reconstructed fields
# Plot the locations of the stations
projection=ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=90.0)
ax_map=fig.add_axes([0.01,0.51,0.98,0.48],projection=projection)
ax_map.set_axis_off()
extent=[-180,180,-90,90]
ax_map.set_extent(extent, crs=projection)
matplotlib.rc('image',aspect='auto')

# Run the fake through the assimilator and produce an iris cube
pm = ic.copy()
result=autoencoder.predict_on_batch(obs_t)
result=tf.reshape(result,[91,180])
pm.data=unnormalise(result)

# Background, grid and land
ax_map.background_patch.set_facecolor((0.88,0.88,0.88,1))
#mg.background.add_grid(ax_map)
land_img_orig=ax_map.background_img(name='GreyT', resolution='low')

# Station locations
mg.observations.plot(ax_map,obs,radius=0.5,
                     facecolor='black',
                     edgecolor='black',
                     linewidths=0.1)
# original pressures as red contours
mg.pressure.plot(ax_map,ic,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,7),
                 colors='red',
                 label=False,
                 linewidths=1)
# Encoded pressures as blue contours
mg.pressure.plot(ax_map,pm,
                 scale=0.01,
                 resolution=0.25,
                 levels=numpy.arange(870,1050,7),
                 colors='blue',
                 label=False,
                 linewidths=1)

mg.utils.plot_label(ax_map,
                    '%04d-%02d-%02d:%02d' % (year,month,day,hour),
                    facecolor=(0.88,0.88,0.88,0.9),
                    fontsize=8,
                    x_fraction=0.98,
                    y_fraction=0.03,
                    verticalalignment='bottom',
                    horizontalalignment='right')

# Scatterplot of encoded v original
ax=fig.add_axes([0.08,0.05,0.45,0.4])
aspect=.225/.4*16/9
# Axes ranges from data
dmin=min(ic.data.min(),pm.data.min())
dmax=max(ic.data.max(),pm.data.max())
dmean=(dmin+dmax)/2
dmax=dmean+(dmax-dmean)*1.05
dmin=dmean-(dmean-dmin)*1.05
if aspect<1:
    ax.set_xlim(dmin/100,dmax/100)
    ax.set_ylim((dmean-(dmean-dmin)*aspect)/100,
                (dmean+(dmax-dmean)*aspect)/100)
else:
    ax.set_ylim(dmin/100,dmax/100)
    ax.set_xlim((dmean-(dmean-dmin)*aspect)/100,
                (dmean+(dmax-dmean)*aspect)/100)
ax.scatter(x=pm.data.flatten()/100,
           y=ic.data.flatten()/100,
           c='black',
           alpha=0.5,
           marker='.',
           s=3)
ax.set(ylabel='20CR2c', 
       xlabel='Constructed')
ax.grid(color='black',
        alpha=0.2,
        linestyle='-', 
        linewidth=0.5)


# Plot the training history
history_save_file=("%s/Machine-Learning-experiments/"+
              "oldstyle_assimilation_1916/"+
              "saved_models/history_to_%04d.pkl") % (
                 os.getenv('SCRATCH'),50)
history=pickle.load( open( history_save_file, "rb" ) )
ax=fig.add_axes([0.62,0.05,0.35,0.4])
# Axes ranges from data
ax.set_xlim(0,len(history['loss']))
ax.set_ylim(0,numpy.max(numpy.concatenate((history['loss'],
                                           history['val_loss']))))
ax.set(xlabel='Epochs', 
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


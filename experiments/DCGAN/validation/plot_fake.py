#!/usr/bin/env python

# Plot global surface weather as faked by the DCGAN

import tensorflow as tf
tf.enable_eager_execution()

import os
import Meteorographica as mg
import iris
import numpy
from pandas import qcut

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs
from matplotlib.patches import Polygon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=False,
                    default=50)
parser.add_argument("--latent-dim", help="No. of latent dimensions",
                    type=int,required=False,
                    default=100)
args = parser.parse_args()

# Load the generator model
model_save_dir=(("%s/Machine-Learning-experiments/"+
                "DCGAN/"+
                "saved_models/Epoch_%04d/generator") % (
                 os.getenv('SCRATCH'),args.epoch))
generator=tf.keras.models.load_model(model_save_dir)

# Make a fake weather field
test_input = tf.random.normal([1, args.latent_dim])
simulation = generator(test_input, training=False)

# Make an Iris cube from the simulated pressure data
cs=iris.coord_systems.RotatedGeogCS(0.0,60.0,270.0)
lat_values=numpy.arange(-90,91,180/78)
latitude=iris.coords.DimCoord(lat_values,
                          standard_name='latitude',
                          units='degrees',
                          coord_system=cs)
lon_values=numpy.arange(-180,181,360/158)
longitude = iris.coords.DimCoord(lon_values,
                                 standard_name='longitude',
                                 units='degrees_east',
                                 coord_system=cs)
dummy_data = numpy.zeros((len(lat_values), len(lon_values)))
dummy_cube = iris.cube.Cube(dummy_data,
                           dim_coords_and_dims=[(latitude, 0),
                                                (longitude, 1)])


# Define the figure (page size, background color, resolution, ...
aspect=16/9.0
fig=Figure(figsize=(10.8*aspect,10.8),     # HD video 
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,                # Don't draw a frame
           subplotpars=None,
           tight_layout=None)
# Attach a canvas
canvas=FigureCanvas(fig)

# All mg plots use Rotated Pole: choose a rotation that shows the global
#  circulation nicely.
projection=ccrs.RotatedPole(pole_longitude=60.0,
                                pole_latitude=0.0,
                                central_rotated_longitude=270.0)
#projection=ccrs.RotatedPole(pole_longitude=180.0,
#                                pole_latitude=90.0,
#                                central_rotated_longitude=0.0)
#projection=ccrs.RotatedPole(pole_longitude=160.0,
#                                pole_latitude=45.0,
#                                central_rotated_longitude=-40.0)

# Define an axes to contain the plot. In this case our axes covers
#  the whole figure
ax = fig.add_axes([0,0,1,1],projection=projection)
ax.set_axis_off() # Don't want surrounding x and y axis
# Set the axes background colour
ax.background_patch.set_facecolor((0.88,0.88,0.88,1))

# Lat and lon range (in rotated-pole coordinates) for plot
extent=[-180.0,180.0,-90.0,90.0]
ax.set_extent(extent, crs=projection)
# Lat:Lon aspect does not match the plot aspect, ignore this and
#  fill the figure with the plot.
matplotlib.rc('image',aspect='auto')

# Add the land
land_img=ax.background_img(name='GreyT', resolution='low')
# Reduce the land contrast
#poly = Polygon(([-180,-90],[180,-90],[180,90],[-180,90]),
#                 facecolor=(0.88,0.88,0.88,0.05))
#ax.add_patch(poly)

# Plot the temperature
t2m=tf.reshape(simulation.numpy()[:,:,:,1],[79,159]).numpy()
t2m *= 50
t2m += 280 # unnormalise
dummy_cube.data=t2m
# Regrid to plot coordinates
plot_cube=mg.utils.dummy_cube(ax,0.25)
t2m = dummy_cube.regrid(plot_cube,iris.analysis.Linear())
# Re-map to highlight small differences
s=t2m.data.shape
t2m.data=qcut(t2m.data.flatten(),20,labels=False,duplicates='drop').reshape(s)
# Plot as a colour map
lats = t2m.coord('latitude').points
lons = t2m.coord('longitude').points
t2m_img=ax.pcolorfast(lons, lats, t2m.data,
                      cmap='coolwarm',
                      vmin=0,
                      vmax=20,
                      alpha=0.5)
   
# Also pressure
prmsl=tf.reshape(simulation.numpy()[:,:,:,0],[79,159]).numpy()
prmsl *= 3000
prmsl += 101325 # unnormalise
dummy_cube.data=prmsl
mg.pressure.plot(ax,dummy_cube,scale=0.01,resolution=0.25,
                 linewidths=1)

# Also precip
prate=tf.reshape(simulation.numpy()[:,:,:,2],[79,159]).numpy()
prate=prate**2
prate /= 10000
dummy_cube.data=prate
mg.precipitation.plot(ax,dummy_cube,resolution=0.25)#,vmin=-0.01,vmax=0.04)

# Add a label showing the date
label="Reconstruction at Epoch %d" % args.epoch
mg.utils.plot_label(ax,label,
                    facecolor=fig.get_facecolor())

# Render the figure as a png
fig.savefig('simulation.png')

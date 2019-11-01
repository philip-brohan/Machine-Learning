#!/usr/bin/env python

# Atmospheric state - near-surface temperature, u-wind, v-wind, and prmsl.
# Show the field associated with a random latent space state.

import os
import IRData.opfc as opfc
import IRData.twcr as twcr
import datetime
import pickle

import tensorflow as tf
tf.enable_eager_execution()

import iris
import numpy
import math

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from pandas import qcut

# Fix dask SPICE bug
import dask
dask.config.set(scheduler='single-threaded')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=True)

args = parser.parse_args()

# Normalisation and un-normalisation functions
def normalise_precip(p):
   res=p.copy()
   res.data = res.data*1000+1.001
   res.data = numpy.log(res.data)
   return res
def unnormalise_precip(p):
   res=p.copy()
   res.data = numpy.exp(res.data)
   res.data = (res.data-1.001)/1000
   return res
def normalise_t2m(p):
   res=p.copy()
   res.data -= 280
   res.data /= 50
   return res
def unnormalise_t2m(p):
   res=p.copy()
   res.data *= 50
   res.data += 280
   return res
def normalise_wind(p):
   res=p.copy()
   res.data /= 12
   return res
def unnormalise_wind(p):
   res=p.copy()
   res.data *= 12
   return res
def normalise_prmsl(p):
   res=p.copy()
   res.data -= 101325
   res.data /= 3000
   return res
def unnormalise_prmsl(p):
   res=p.copy()
   res.data *= 3000
   res.data += 101325
   return res
# Normalise temperature by quantiles - just for plotting - balances colours
def quantile_t2m(p):
   res=p.copy()
   res.data[res.data>300.10]=0.95
   res.data[res.data>299.9]=0.90
   res.data[res.data>298.9]=0.85
   res.data[res.data>297.5]=0.80
   res.data[res.data>295.7]=0.75
   res.data[res.data>293.5]=0.70
   res.data[res.data>290.1]=0.65
   res.data[res.data>287.6]=0.60
   res.data[res.data>283.7]=0.55
   res.data[res.data>280.2]=0.50
   res.data[res.data>277.2]=0.45
   res.data[res.data>274.4]=0.40
   res.data[res.data>272.3]=0.35
   res.data[res.data>268.3]=0.30
   res.data[res.data>261.4]=0.25
   res.data[res.data>254.6]=0.20
   res.data[res.data>249.1]=0.15
   res.data[res.data>244.9]=0.10
   res.data[res.data>240.5]=0.05
   res.data[res.data>0.95]=0.0
   return res

# Projection for tensors and plotting
cs=iris.coord_systems.RotatedGeogCS(90,180,0)

# Define a dummy cube to load with the compressed data
def dummy_cube():
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
    return(dummy_cube)


# Load the latent-space representation, and convert it back into normal space
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_uk_centred_var_insol/saved_models/"+
                  "Epoch_%04d/generator") % (
                      os.getenv('SCRATCH'),args.epoch)
generator=tf.keras.models.load_model(model_save_file,compile=False)

# Random latent state
ls=tf.convert_to_tensor(numpy.random.normal(size=100),numpy.float32)
ls = tf.reshape(ls,[1,100])
result=generator.predict_on_batch(ls)
result = tf.reshape(result,[79,159,4])
t2m=dummy_cube()
t2m.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
t2m = unnormalise_t2m(t2m)
prmsl=dummy_cube()
prmsl.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
prmsl = unnormalise_prmsl(prmsl)
u10m=dummy_cube()
u10m.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
u10m = unnormalise_wind(u10m)
v10m=dummy_cube()
v10m.data = tf.reshape(result.numpy()[:,:,3],[79,159]).numpy()
v10m = unnormalise_wind(v10m)
    
mask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('DATADIR'))

# Define the figure (page size, background color, resolution, ...
fig=Figure(figsize=(19.2,10.8),              # Width, Height (inches)
           dpi=100,
           facecolor=(0.5,0.5,0.5,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,                # Don't draw a frame
           subplotpars=None,
           tight_layout=None)
fig.set_frameon(False) 
# Attach a canvas
canvas=FigureCanvas(fig)

# Projection for plotting
cs=iris.coord_systems.RotatedGeogCS(90,180,0)

def plot_cube(resolution,xmin=-180,xmax=180,ymin=-90,ymax=90):

    lat_values=numpy.arange(ymin,ymax+resolution,resolution)
    latitude = iris.coords.DimCoord(lat_values,
                                    standard_name='latitude',
                                    units='degrees_north',
                                    coord_system=cs)
    lon_values=numpy.arange(xmin,xmax+resolution,resolution)
    longitude = iris.coords.DimCoord(lon_values,
                                     standard_name='longitude',
                                     units='degrees_east',
                                     coord_system=cs)
    dummy_data = numpy.zeros((len(lat_values), len(lon_values)))
    plot_cube = iris.cube.Cube(dummy_data,
                               dim_coords_and_dims=[(latitude, 0),
                                                    (longitude, 1)])
    return plot_cube

# Make the wind noise
def wind_field(uw,vw,z,sequence=None,iterations=50,epsilon=0.003,sscale=1):
    (width,height)=z.data.shape
    # Each point in this field has an index location (i,j)
    #  and a real (x,y) position
    xmin=numpy.min(uw.coords()[0].points)
    xmax=numpy.max(uw.coords()[0].points)
    ymin=numpy.min(uw.coords()[1].points)
    ymax=numpy.max(uw.coords()[1].points)
    # Convert between index and real positions
    def i_to_x(i):
        return xmin + (i/width) * (xmax-xmin)
    def j_to_y(j):
        return ymin + (j/height) * (ymax-ymin)
    def x_to_i(x):
        return numpy.minimum(width-1,numpy.maximum(0, 
                numpy.floor((x-xmin)/(xmax-xmin)*(width-1)))).astype(int)
    def y_to_j(y):
        return numpy.minimum(height-1,numpy.maximum(0, 
                numpy.floor((y-ymin)/(ymax-ymin)*(height-1)))).astype(int)
    i,j=numpy.mgrid[0:width,0:height]
    x=i_to_x(i)
    y=j_to_y(j)
    # Result is a distorted version of the random field
    result=z.copy()
    # Repeatedly, move the x,y points according to the vector field
    #  and update result with the random field at their locations
    ss=uw.copy()
    ss.data=numpy.sqrt(uw.data**2+vw.data**2)
    if sequence is not None:
        startsi=numpy.arange(0,iterations,3)
        endpoints=numpy.tile(startsi,1+(width*height)//len(startsi))
        endpoints += sequence%iterations
        endpoints[endpoints>=iterations] -= iterations
        startpoints=endpoints-25
        startpoints[startpoints<0] += iterations
        endpoints=endpoints[0:(width*height)].reshape(width,height)
        startpoints=startpoints[0:(width*height)].reshape(width,height)
    else:
        endpoints=iterations+1 
        startpoints=-1       
    for k in range(iterations):
        x += epsilon*vw.data[i,j]
        x[x>xmax]=xmax
        x[x<xmin]=xmin
        y += epsilon*uw.data[i,j]
        y[y>ymax]=y[y>ymax]-ymax+ymin
        y[y<ymin]=y[y<ymin]-ymin+ymax
        i=x_to_i(x)
        j=y_to_j(y)
        update=z.data*ss.data/sscale
        update[(endpoints>startpoints) & ((k>endpoints) | (k<startpoints))]=0
        update[(startpoints>endpoints) & ((k>endpoints) & (k<startpoints))]=0
        result.data[i,j] += update
    return result

wind_pc=plot_cube(0.2,-180,180,
                      -90,90)   
rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
z=mask.regrid(u10m,iris.analysis.Linear())
(width,height)=z.data.shape
z.data=numpy.random.rand(width,height)
wind_noise_field=wind_field(u10m,v10m,z,sequence=None,epsilon=0.01)

# Define an axes to contain the plot. In this case our axes covers
#  the whole figure
ax = fig.add_axes([0,0,1,1])
ax.set_axis_off() # Don't want surrounding x and y axis

# Lat and lon range (in rotated-pole coordinates) for plot
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_aspect('auto')

# Background
ax.add_patch(Rectangle((0,0),1,1,facecolor=(0.6,0.6,0.6,1),fill=True,zorder=1))

# Draw lines of latitude and longitude
for lat in range(-90,95,5):
    lwd=0.75
    x=[]
    y=[]
    for lon in range(-180,181,1):
        rp=iris.analysis.cartography.rotate_pole(numpy.array(lon),
                                                 numpy.array(lat),
                                                 180,
                                                 90)
        nx=rp[0]
        if nx>180: nx -= 360
        ny=rp[1]
        if(len(x)==0 or (abs(nx-x[-1])<10 and abs(ny-y[-1])<10)):
            x.append(nx)
            y.append(ny)
        else:
            ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                               zorder=10))
            x=[]
            y=[]
    if(len(x)>1):        
        ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                           zorder=10))

for lon in range(-180,185,5):
    lwd=0.75
    x=[]
    y=[]
    for lat in range(-90,90,1):
        rp=iris.analysis.cartography.rotate_pole(numpy.array(lon),
                                                 numpy.array(lat),
                                                 180,
                                                 90)
        nx=rp[0]
        if nx>180: nx -= 360
        ny=rp[1]
        if(len(x)==0 or (abs(nx-x[-1])<10 and abs(ny-y[-1])<10)):
            x.append(nx)
            y.append(ny)
        else:
            ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                               zorder=10))
            x=[]
            y=[]
    if(len(x)>1):        
        ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                           zorder=10))

# Plot the land mask
mask_pc=plot_cube(0.05,-180,180,-90,90)   
mask = mask.regrid(mask_pc,iris.analysis.Linear())
lats = mask.coord('latitude').points
lons = mask.coord('longitude').points
mask_img = ax.pcolorfast(lons, lats, mask.data,
                         cmap=matplotlib.colors.ListedColormap(
                                ((0.4,0.4,0.4,0),
                                 (0.4,0.4,0.4,1))),
                         vmin=0,
                         vmax=1,
                         alpha=1.0,
                         zorder=20)


# Plot the T2M
t2m_pc=plot_cube(0.05,-180,180,
                      -90,90)   
t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
t2m=quantile_t2m(t2m)
# Adjust to show the wind
wscale=200
s=wind_noise_field.data.shape
wind_noise_field.data=qcut(wind_noise_field.data.flatten(),wscale,labels=False,
                             duplicates='drop').reshape(s)-(wscale-1)/2

# Plot as a colour map
wnf=wind_noise_field.regrid(t2m,iris.analysis.Linear())
t2m_img = ax.pcolorfast(lons, lats, t2m.data*800+wnf.data,
                        cmap='RdYlBu_r',
                        alpha=0.8,
                        zorder=100)

# PRMSL contours
prmsl_pc=plot_cube(0.25,-180,180,
                         -90,90)   
prmsl = prmsl.regrid(prmsl_pc,iris.analysis.Linear())
lats = prmsl.coord('latitude').points
lons = prmsl.coord('longitude').points
lons,lats = numpy.meshgrid(lons,lats)
CS=ax.contour(lons, lats, prmsl.data*0.01,
                           colors='black',
                           linewidths=1.0,
                           alpha=1.0,
                           levels=numpy.arange(870,1050,10),
                           zorder=200)

# Overlay the latent-space representation in the SE Pacific
ax2=fig.add_axes([0.025,0.05,0.15,0.15*16/9])
ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
ax2.set_axis_off() # Don't want surrounding x and y axis
x=numpy.linspace(0,10,10)
latent_img = ax2.pcolorfast(x,x,ls.numpy().reshape(10,10),
                           cmap='viridis',
                             alpha=1.0,
                             vmin=-3,
                             vmax=3,
                             zorder=1000)

# Render the figure as a png
fig.savefig('random_ls.png')

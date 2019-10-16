#!/usr/bin/env python

# Model training results plot

import tensorflow as tf
tf.enable_eager_execution()
import numpy

import IRData.twcr as twcr
import IRData.opfc as opfc
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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=True)
args = parser.parse_args()

# Projection for tensors and plotting
cs=iris.coord_systems.RotatedGeogCS(90,180,0)

# Define cube for regridding data to match the training tensors
def tensor_cube(cbe):
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

# Define cube for making plots
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
    # Each point in this field has an index location (i,j)
    #  and a real (x,y) position
    (width,height)=z.data.shape
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


# Function to do the multivariate plot
lsmask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('DATADIR'))
# Random field for the wind noise
z=lsmask.regrid(plot_cube(0.5),iris.analysis.Linear())
(width,height)=z.data.shape
z.data=numpy.random.rand(width,height)
def three_plot(ax,t2m,u10m,v10m,prmsl):
    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
    ax.set_aspect('auto')
    ax.set_axis_off() # Don't want surrounding x and y axis
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
            nx=rp[0]+0
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
            nx=rp[0]+0
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
    # Add the continents
    mask_pc = plot_cube(0.05)   
    lsmask = iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('DATADIR'))
    lsmask = lsmask.regrid(mask_pc,iris.analysis.Linear())
    lats = lsmask.coord('latitude').points
    lons = lsmask.coord('longitude').points
    mask_img = ax.pcolorfast(lons, lats, lsmask.data,
                             cmap=matplotlib.colors.ListedColormap(
                                    ((0.4,0.4,0.4,0),
                                     (0.4,0.4,0.4,1))),
                             vmin=0,
                             vmax=1,
                             alpha=1.0,
                             zorder=20)
    
    # Calculate the wind noise
    wind_pc=plot_cube(0.5)   
    rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
    u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
    v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
    wind_noise_field=wind_field(u10m,v10m,z,sequence=None,epsilon=0.01)

    # Plot the temperature
    t2m_pc=plot_cube(0.05)   
    t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
    t2m=quantile_t2m(t2m)
    # Adjust to show the wind
    wscale=200
    s=wind_noise_field.data.shape
    wind_noise_field.data=qcut(wind_noise_field.data.flatten(),wscale,labels=False,
                                 duplicates='drop').reshape(s)-(wscale-1)/2

    # Plot as a colour map
    wnf=wind_noise_field.regrid(t2m,iris.analysis.Linear())
    t2m_img = ax.pcolorfast(lons, lats, t2m.data*200+wnf.data,
                            cmap='RdYlBu_r',
                            alpha=0.8,
                            zorder=100)

    # Plot the prmsl
    prmsl_pc=plot_cube(0.25)   
    prmsl = prmsl.regrid(prmsl_pc,iris.analysis.Linear())
    lats = prmsl.coord('latitude').points
    lons = prmsl.coord('longitude').points
    lons,lats = numpy.meshgrid(lons,lats)
    CS=ax.contour(lons, lats, prmsl.data*0.01,
                               colors='black',
                               linewidths=0.5,
                               alpha=1.0,
                               levels=numpy.arange(870,1050,10),
                               zorder=200)
   
 
# Load the validation data
prmsl=twcr.load('prmsl',datetime.datetime(2010,3,12,18),
                           version='2c')
prmsl=tensor_cube(prmsl.extract(iris.Constraint(member=1)))
t2m=twcr.load('air.2m',datetime.datetime(2010,3,12,18),
                           version='2c')
t2m=tensor_cube(t2m.extract(iris.Constraint(member=1)))
u10m=twcr.load('uwnd.10m',datetime.datetime(2010,3,12,18),
                           version='2c')
u10m=tensor_cube(u10m.extract(iris.Constraint(member=1)))
v10m=twcr.load('vwnd.10m',datetime.datetime(2010,3,12,18),
                           version='2c')
v10m=tensor_cube(v10m.extract(iris.Constraint(member=1)))

# Convert the validation data into tensor format
t2m_t = tf.convert_to_tensor(normalise_t2m(t2m).data,numpy.float32)
t2m_t = tf.reshape(t2m_t,[79,159,1])
prmsl_t = tf.convert_to_tensor(normalise_prmsl(prmsl).data,numpy.float32)
prmsl_t = tf.reshape(prmsl_t,[79,159,1])
u10m_t = tf.convert_to_tensor(normalise_wind(u10m).data,numpy.float32)
u10m_t = tf.reshape(u10m_t,[79,159,1])
v10m_t = tf.convert_to_tensor(normalise_wind(v10m).data,numpy.float32)
v10m_t = tf.reshape(v10m_t,[79,159,1])

# Get autoencoded versions of the validation data
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_uk_centred/saved_models/"+
                  "Epoch_%04d/autoencoder") % (
                      os.getenv('SCRATCH'),args.epoch)
autoencoder=tf.keras.models.load_model(model_save_file,compile=False)
ict = tf.concat([t2m_t,prmsl_t,u10m_t,v10m_t],2) # Now [79,159,4]
ict = tf.reshape(ict,[1,79,159,4])
result = autoencoder.predict_on_batch(ict)
result = tf.reshape(result,[79,159,4])

# Convert the encoded fields back to unnormalised cubes 
t2m_r=t2m.copy()
t2m_r.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
t2m_r = unnormalise_t2m(t2m_r)
#t2m_r = unnormalise_t2m(normalise_t2m(t2m))
prmsl_r=prmsl.copy()
prmsl_r.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
prmsl_r = unnormalise_prmsl(prmsl_r)
#prate_r = unnormalise_precip(normalise_precip(prate))
u10m_r=u10m.copy()
u10m_r.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
u10m_r = unnormalise_wind(u10m_r)
#u10m_r = unnormalise_wind(normalise_wind(u10m))
v10m_r=v10m.copy()
v10m_r.data = tf.reshape(result.numpy()[:,:,3],[79,159]).numpy()
v10m_r = unnormalise_wind(v10m_r)
#v10m_r = unnormalise_wind(normalise_wind(v10m))

# Plot the two fields and a scatterplot for each variable
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
ax_original=fig.add_axes([0.005,0.525,0.75,0.45])
three_plot(ax_original,t2m,u10m,v10m,prmsl)
ax_reconstructed=fig.add_axes([0.005,0.025,0.75,0.45])
three_plot(ax_reconstructed,t2m_r,u10m_r,v10m_r,prmsl_r)

# Scatterplot of encoded v original
def plot_scatter(ax,ic,pm):
    dmin=min(ic.min(),pm.min())
    dmax=max(ic.max(),pm.max())
    dmean=(dmin+dmax)/2
    dmax=dmean+(dmax-dmean)*1.02
    dmin=dmean-(dmean-dmin)*1.02
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
    
ax_t2m=fig.add_axes([0.83,0.80,0.16,0.17])
plot_scatter(ax_t2m,t2m.data,t2m_r.data)
ax_prmsl=fig.add_axes([0.83,0.55,0.16,0.17])
plot_scatter(ax_prmsl,prmsl.data*0.01,prmsl_r.data*0.01)
ax_u10m=fig.add_axes([0.83,0.30,0.16,0.17])
plot_scatter(ax_u10m,u10m.data,u10m_r.data)
ax_v10m=fig.add_axes([0.83,0.05,0.16,0.17])
plot_scatter(ax_v10m,v10m.data,v10m_r.data)

# Render the figure as a png
fig.savefig("comparison_results.png")


#!/usr/bin/env python

# Global surface weather plot

# Set up the figure and add the continents as background
# Overlay multivariate weather: pressure, temperature and precip.

import os
import IRData.twcr as twcr
import Meteorographica as mg
import datetime
import iris
import numpy
from pandas import qcut

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs
from matplotlib.patches import Polygon

# Get the datetime to plot from commandline arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year",
                    type=int,required=True)
parser.add_argument("--month", help="Integer month",
                    type=int,required=True)
parser.add_argument("--day", help="Day of month",
                    type=int,required=True)
parser.add_argument("--hour", help="Time of day (0 to 23.99)",
                    type=float,required=True)
parser.add_argument("--opdir", help="Directory for output files",
                    default=".",
                    type=str,required=False)
args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

dte=datetime.datetime(args.year,args.month,args.day,
                      int(args.hour),int(args.hour%1*60))


# Define the figure (page size, background color, resolution, ...
aspect=16/9.0
fig=Figure(figsize=(10.8*aspect,10.8),              # Width, Height (inches)
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
t2m=twcr.load('air.2m',dte,version='2c')
t2m=t2m.extract(iris.Constraint(member=1))
# Regrid to plot coordinates
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
prmsl=twcr.load('prmsl',dte,version='2c')
prmsl=prmsl.extract(iris.Constraint(member=1))
mg.pressure.plot(ax,prmsl,scale=0.01,resolution=0.25,
                 linewidths=1,label=False)

# Also precip
prate=twcr.load('prate',dte,version='2c')
prate=prate.extract(iris.Constraint(member=1))
mg.precipitation.plot(ax,prate,resolution=0.25,vmin=-0.01,vmax=0.04)

# Add a label showing the date
label='%04d-%02d-%02d:%02d' % (args.year,args.month,
                                       args.day,int(args.hour))
mg.utils.plot_label(ax,label,
                    facecolor=fig.get_facecolor())

# Render the figure as a png
fig.savefig('%s/V2c_%04d%02d%02d%02d%02d.png' % 
               (args.opdir,args.year,args.month,args.day,
                           int(args.hour),int(args.hour%1*60)))

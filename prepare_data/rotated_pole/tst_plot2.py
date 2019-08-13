#!/usr/bin/env python

# Meteorographica example script

# Set up the figure and add the continents as background
# Overlay multivariate weather: pressure, wind and precip.

import Meteorographica as mg
import iris
import numpy

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cartopy
import cartopy.crs as ccrs

import pkg_resources
import gzip
import pickle

# Define the figure (page size, background color, resolution, ...
aspect=16/9.0
fig=Figure(figsize=(22,22/aspect),              # Width, Height (inches)
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,                # Don't draw a frame
           subplotpars=None,
           tight_layout=None)
# Attach a canvas
canvas=FigureCanvas(fig)

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

# All mg plots use Rotated Pole: choose a rotation that shows the global
#  circulation nicely.
projection=ccrs.RotatedPole(pole_longitude=60.0,
                                pole_latitude=0.0,
                                central_rotated_longitude=270.0)

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

# Draw a lat:lon grid
mg.background.add_grid(ax,
                       sep_major=5,
                       sep_minor=2.5,
                       color=(0,0.3,0,0.2))


# Add the land
land_img=ax.background_img(name='GreyT', resolution='low')

# Also pressure
edf=pkg_resources.resource_filename(
      pkg_resources.Requirement.parse('Meteorographica'),
                 'example_data/20CR2c.1987101606.prmsl.nc')
prmsl=rr_cube(iris.load_cube(edf))
prmsl=prmsl.extract(iris.Constraint(member=1))
mg.pressure.plot(ax,prmsl,scale=0.01,resolution=0.25)

# Also precip
edf=pkg_resources.resource_filename(
      pkg_resources.Requirement.parse('Meteorographica'),
                 'example_data/20CR2c.1987101606.prate.nc')
prate=rr_cube(iris.load_cube(edf))
prate=prate.extract(iris.Constraint(member=1))
mg.precipitation.plot(ax,prate,resolution=0.25)

# Add the observations 
edf=pkg_resources.resource_filename(
      pkg_resources.Requirement.parse('Meteorographica'),
                 'example_data/20CR2c.1987101606.observations.pklz')
of=gzip.open(edf,'rb')
obs=pickle.load(of,encoding='latin1')
of.close()
mg.observations.plot(ax,obs,radius=0.25)

# Add a label showing the date
label="16th October 1987 at 06 GMT"
mg.utils.plot_label(ax,label,
                    facecolor=fig.get_facecolor())

# Render the figure as a png
fig.savefig('tst2.png')

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
def three_plot(ax,t2m,u10m,v10m,precip):
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

    # Plot the precip
    precip_pc=plot_cube(0.25)   
    precip = precip.regrid(precip_pc,iris.analysis.Linear())
    precip=normalise_precip(precip)
    wnf=wind_noise_field.regrid(precip,iris.analysis.Linear())
    precip.data += wnf.data/1000
    cols=[]
    for ci in range(10):
        cols.append([0.06885643,0.14208946,0.07903363,0.0])
    for ci in range(50):
        cols.append([0.06885643,0.14208946,0.07903363,ci/50])
    cm_data = [[ 0.06885643, 0.14208946, 0.07903363],
               [ 0.07022733, 0.145481  , 0.08182252],
               [ 0.07158041, 0.14886551, 0.08459735],
               [ 0.07289843, 0.15224692, 0.08735815],
               [ 0.07419261, 0.15562329, 0.09010536],
               [ 0.07545752, 0.15899606, 0.0928391 ],
               [ 0.07669294, 0.16236561, 0.09555954],
               [ 0.0779044 , 0.16573113, 0.09826708],
               [ 0.07908113, 0.16909506, 0.10096151],
               [ 0.08023865, 0.17245461, 0.10364358],
               [ 0.0813567 , 0.17581401, 0.1063126 ],
               [ 0.08245885, 0.17916893, 0.10896978],
               [ 0.08351901, 0.18252464, 0.11161396],
               [ 0.08456274, 0.18587646, 0.1142466 ],
               [ 0.08556724, 0.18922894, 0.11686652],
               [ 0.08655243, 0.19257854, 0.119475  ],
               [ 0.08750046, 0.19592877, 0.12207102],
               [ 0.08842694, 0.19927694, 0.12465567],
               [ 0.08931759, 0.20262583, 0.12722804],
               [ 0.09018515, 0.2059733 , 0.1297891 ],
               [ 0.09101742, 0.20932169, 0.13233799],
               [ 0.09182583, 0.21266913, 0.13487564],
               [ 0.09259864, 0.21601782, 0.13740115],
               [ 0.09334765, 0.21936582, 0.13991552],
               [ 0.09405985, 0.22271554, 0.14241766],
               [ 0.09474918, 0.22606468, 0.14490881],
               [ 0.09539954, 0.22941612, 0.14738752],
               [ 0.0960289 , 0.23276689, 0.14985548],
               [ 0.09661612, 0.23612068, 0.15231064],
               [ 0.09718523, 0.23947356, 0.15475537],
               [ 0.09770794, 0.2428303 , 0.1571868 ],
               [ 0.09821309, 0.24618625, 0.15960787],
               [ 0.09867327, 0.24954595, 0.16201567],
               [ 0.09911166, 0.25290573, 0.16441267],
               [ 0.09951033, 0.25626852, 0.16679686],
               [ 0.09988043, 0.25963262, 0.16916945],
               [ 0.1002173 , 0.26299882, 0.17152986],
               [ 0.10051752, 0.2663677 , 0.17387765],
               [ 0.10079233, 0.2697376 , 0.17621409],
               [ 0.10102103, 0.27311168, 0.17853665],
               [ 0.10122817, 0.27648631, 0.18084823],
               [ 0.10138903, 0.2798652 , 0.18314573],
               [ 0.10152195, 0.28324567, 0.18543137],
               [ 0.10161961, 0.28662882, 0.18770415],
               [ 0.10167654, 0.29001546, 0.18996328],
               [ 0.10170933, 0.29340326, 0.19221085],
               [ 0.10169001, 0.29679616, 0.19444309],
               [ 0.1016439 , 0.30019064, 0.19666326],
               [ 0.10156046, 0.30358817, 0.19886988],
               [ 0.10143361, 0.30698958, 0.201062  ],
               [ 0.10127976, 0.31039267, 0.20324181],
               [ 0.1010766 , 0.31380039, 0.20540608],
               [ 0.10083896, 0.31721078, 0.20755678],
               [ 0.10057112, 0.32062332, 0.20969446],
               [ 0.10024781, 0.32404118, 0.21181533],
               [ 0.09989443, 0.32746115, 0.21392297],
               [ 0.0995047 , 0.33088402, 0.21601636],
               [ 0.09906254, 0.3343117 , 0.218093  ],
               [ 0.0985884 , 0.33774172, 0.2201558 ],
               [ 0.09807546, 0.34117488, 0.22220362],
               [ 0.09750943, 0.34461278, 0.22423417],
               [ 0.09690948, 0.3480532 , 0.22625023],
               [ 0.09627239, 0.35149649, 0.22825119],
               [ 0.09557752, 0.35494489, 0.23023372],
               [ 0.09484678, 0.35839595, 0.23220107],
               [ 0.09407947, 0.36184972, 0.23415298],
               [ 0.09325665, 0.36530817, 0.23608636],
               [ 0.09239033, 0.36876996, 0.23800293],
               [ 0.09148554, 0.37223456, 0.23990332],
               [ 0.09053798, 0.37570238, 0.2417867 ],
               [ 0.08953167, 0.3791749 , 0.24365037],
               [ 0.08848513, 0.38265026, 0.2454971 ],
               [ 0.08739776, 0.3861285 , 0.24732662],
               [ 0.08626466, 0.38961004, 0.24913796],
               [ 0.08507286, 0.39309597, 0.25092886],
               [ 0.08383877, 0.39658475, 0.25270175],
               [ 0.0825619 , 0.40007639, 0.25445635],
               [ 0.08124183, 0.4035709 , 0.25619236],
               [ 0.07986463, 0.40706939, 0.25790734],
               [ 0.07843839, 0.41057109, 0.25960234],
               [ 0.07696833, 0.41407555, 0.26127794],
               [ 0.07545435, 0.41758275, 0.26293384],
               [ 0.07389642, 0.42109266, 0.26456976],
               [ 0.07228966, 0.42460561, 0.26618462],
               [ 0.07062863, 0.42812193, 0.26777721],
               [ 0.06892492, 0.43164076, 0.26934902],
               [ 0.06717918, 0.43516204, 0.27089974],
               [ 0.06539233, 0.43868571, 0.27242909],
               [ 0.06356557, 0.44221171, 0.27393677],
               [ 0.06170041, 0.44573995, 0.27542249],
               [ 0.05979877, 0.44927035, 0.27688596],
               [ 0.05785702, 0.45280318, 0.27832599],
               [ 0.05588066, 0.45633814, 0.27974269],
               [ 0.05387746, 0.45987489, 0.28113643],
               [ 0.05185174, 0.4634133 , 0.28250695],
               [ 0.04980869, 0.46695324, 0.28385397],
               [ 0.04775453, 0.47049459, 0.28517723],
               [ 0.04569662, 0.47403719, 0.28647649],
               [ 0.04364369, 0.47758089, 0.28775151],
               [ 0.04160606, 0.48112551, 0.28900203],
               [ 0.0395867 , 0.48467089, 0.29022786],
               [ 0.03763814, 0.48821682, 0.29142876],
               [ 0.03580155, 0.49176311, 0.29260454],
               [ 0.03408899, 0.49530953, 0.29375502],
               [ 0.03251323, 0.49885586, 0.29488003],
               [ 0.03108771, 0.50240186, 0.29597942],
               [ 0.02982666, 0.50594726, 0.29705307],
               [ 0.02874508, 0.50949179, 0.29810087],
               [ 0.02785177, 0.51303546, 0.29912154],
               [ 0.02716441, 0.51657792, 0.3001151 ],
               [ 0.02670582, 0.52011863, 0.30108235],
               [ 0.02649479, 0.52365722, 0.30202332],
               [ 0.02655111, 0.52719334, 0.30293807],
               [ 0.0268957 , 0.53072659, 0.30382669],
               [ 0.02754542, 0.53425678, 0.30468824],
               [ 0.02851433, 0.53778386, 0.30552072],
               [ 0.02984163, 0.54130675, 0.30632735],
               [ 0.03155357, 0.54482496, 0.30710846],
               [ 0.03367174, 0.54833822, 0.3078629 ],
               [ 0.03621581, 0.55184641, 0.30858828],
               [ 0.03923368, 0.55534819, 0.3092893 ],
               [ 0.04267874, 0.55884313, 0.30996532],
               [ 0.04642777, 0.56233118, 0.31061247],
               [ 0.05049037, 0.56581074, 0.31123743],
               [ 0.05483677, 0.56928169, 0.31183596],
               [ 0.05945722, 0.57274288, 0.31241135],
               [ 0.06433628, 0.57619346, 0.3129641 ],
               [ 0.0694603 , 0.57963256, 0.31349439],
               [ 0.07481987, 0.58305904, 0.31400453],
               [ 0.08040588, 0.58647176, 0.31449636],
               [ 0.08620977, 0.58986967, 0.3149695 ],
               [ 0.09222724, 0.59325117, 0.31543058],
               [ 0.0984523 , 0.59661516, 0.31587743],
               [ 0.10488194, 0.59995997, 0.31631578],
               [ 0.11151261, 0.60328396, 0.31675059],
               [ 0.11834151, 0.60658549, 0.31718585],
               [ 0.12536778, 0.6098628 , 0.31762468],
               [ 0.13258816, 0.61311399, 0.31807501],
               [ 0.13999895, 0.6163372 , 0.31854464],
               [ 0.14759587, 0.61953055, 0.31904185],
               [ 0.15537304, 0.62269226, 0.31957617],
               [ 0.16332236, 0.62582066, 0.32015839],
               [ 0.17143307, 0.62891434, 0.32080039],
               [ 0.17969137, 0.63197223, 0.32151494],
               [ 0.18808025, 0.63499369, 0.32231537],
               [ 0.19658051, 0.63797853, 0.32321469],
               [ 0.20516717, 0.6409274 , 0.32422681],
               [ 0.21381461, 0.64384147, 0.32536378],
               [ 0.22249597, 0.64672254, 0.32663605],
               [ 0.23118418, 0.64957297, 0.32805203],
               [ 0.23985293, 0.65239558, 0.32961779],
               [ 0.24847602, 0.65519373, 0.33133728],
               [ 0.25703342, 0.65797057, 0.33321113],
               [ 0.26550717, 0.66072943, 0.3352381 ],
               [ 0.2738829 , 0.66347352, 0.33741504],
               [ 0.28214978, 0.66620584, 0.33973735],
               [ 0.29030037, 0.66892916, 0.34219931],
               [ 0.29833024, 0.67164589, 0.34479452],
               [ 0.30623756, 0.67435818, 0.34751622],
               [ 0.31401727, 0.67706862, 0.35035755],
               [ 0.32167446, 0.67977832, 0.35331149],
               [ 0.32921313, 0.68248841, 0.35637146],
               [ 0.33663762, 0.6851999 , 0.35953132],
               [ 0.34394207, 0.68791541, 0.36278437],
               [ 0.35114051, 0.69063416, 0.36612568],
               [ 0.35823533, 0.69335721, 0.36954994],
               [ 0.36522563, 0.69608614, 0.37305153],
               [ 0.37212492, 0.69881984, 0.37662736],
               [ 0.37892642, 0.70156084, 0.38027132],
               [ 0.3856453 , 0.70430747, 0.38398168],
               [ 0.39227735, 0.70706166, 0.38775317],
               [ 0.39883103, 0.70982284, 0.39158356],
               [ 0.40531204, 0.71259083, 0.39547048],
               [ 0.41171522, 0.7153677 , 0.39940867],
               [ 0.41805337, 0.71815166, 0.40339835],
               [ 0.42432867, 0.72094309, 0.40743704],
               [ 0.43053587, 0.72374398, 0.41151964],
               [ 0.43668612, 0.72655264, 0.41564705],
               [ 0.44278167, 0.72936931, 0.41981738],
               [ 0.44882469, 0.73219416, 0.42402884],
               [ 0.45481188, 0.73502867, 0.42827737],
               [ 0.46075114, 0.73787164, 0.43256386],
               [ 0.46664458, 0.74072315, 0.43688699],
               [ 0.47249405, 0.74358334, 0.44124545],
               [ 0.47830134, 0.74645233, 0.44563802],
               [ 0.48406816, 0.74933022, 0.45006358],
               [ 0.48979527, 0.75221736, 0.45452056],
               [ 0.49548385, 0.75511394, 0.45900769],
               [ 0.50113714, 0.75801961, 0.46352502],
               [ 0.50675657, 0.76093444, 0.46807172],
               [ 0.51234353, 0.76385852, 0.47264701],
               [ 0.51789934, 0.76679191, 0.47725017],
               [ 0.52342527, 0.76973468, 0.48188052],
               [ 0.52892256, 0.77268689, 0.48653744],
               [ 0.53439237, 0.7756486 , 0.49122035],
               [ 0.53983582, 0.77861987, 0.4959287 ],
               [ 0.54525401, 0.78160076, 0.500662  ],
               [ 0.55064796, 0.78459132, 0.50541978],
               [ 0.55601867, 0.78759161, 0.5102016 ],
               [ 0.56136709, 0.79060167, 0.51500705],
               [ 0.56669416, 0.79362156, 0.51983577],
               [ 0.57200073, 0.79665133, 0.5246874 ],
               [ 0.57728767, 0.79969103, 0.52956161],
               [ 0.5825556 , 0.80274077, 0.53445795],
               [ 0.587804  , 0.80580103, 0.53937491],
               [ 0.59303502, 0.80887141, 0.54431345],
               [ 0.59824938, 0.81195197, 0.54927334],
               [ 0.60344779, 0.81504274, 0.55425434],
               [ 0.60863092, 0.8181438 , 0.55925625],
               [ 0.61379944, 0.82125518, 0.56427887],
               [ 0.61895395, 0.82437694, 0.56932201],
               [ 0.62409506, 0.82750913, 0.57438551],
               [ 0.62922216, 0.83065222, 0.5794679 ],
               [ 0.63433574, 0.83380631, 0.58456889],
               [ 0.63943758, 0.83697102, 0.58968966],
               [ 0.64452821, 0.84014641, 0.59483008],
               [ 0.64960814, 0.84333252, 0.59999004],
               [ 0.65467785, 0.84652942, 0.60516941],
               [ 0.65973664, 0.8497376 , 0.61036667],
               [ 0.66478454, 0.8529573 , 0.61558114],
               [ 0.66982361, 0.85618797, 0.62081462],
               [ 0.67485427, 0.85942969, 0.62606703],
               [ 0.67987695, 0.8626825 , 0.6313383 ],
               [ 0.68489103, 0.86594688, 0.63662701],
               [ 0.68989598, 0.86922324, 0.64193178],
               [ 0.69489414, 0.87251091, 0.64725509],
               [ 0.69988588, 0.87580994, 0.65259689],
               [ 0.70487156, 0.87912041, 0.6579571 ],
               [ 0.70984902, 0.88244339, 0.6633321 ],
               [ 0.71482059, 0.88577817, 0.66872453],
               [ 0.71978715, 0.88912459, 0.67413514],
               [ 0.72474903, 0.89248272, 0.67956388],
               [ 0.72970443, 0.89585351, 0.68500748],
               [ 0.73465479, 0.89923659, 0.69046752],
               [ 0.73960143, 0.90263159, 0.69594546],
               [ 0.74454464, 0.90603859, 0.70144129],
               [ 0.74948217, 0.90945876, 0.70695082],
               [ 0.75441627, 0.91289135, 0.71247711],
               [ 0.75934783, 0.91633616, 0.71802108],
               [ 0.76427661, 0.91979348, 0.72358186],
               [ 0.76920074, 0.92326438, 0.72915567],
               [ 0.77412316, 0.92674772, 0.73474699],
               [ 0.77904414, 0.93024359, 0.74035579],
               [ 0.78396177, 0.93375306, 0.74597816],
               [ 0.78887747, 0.93727567, 0.75161605],
               [ 0.79379251, 0.94081106, 0.75727124],
               [ 0.79870604, 0.9443598 , 0.76294165],
               [ 0.80361734, 0.94792244, 0.76862535],
               [ 0.80852873, 0.9514981 , 0.77432618],
               [ 0.81344004, 0.95508703, 0.7800433 ],
               [ 0.81834917, 0.95869047, 0.78577205],
               [ 0.82325913, 0.96230716, 0.79151775],
               [ 0.82817006, 0.96593722, 0.79728024],
               [ 0.8330792 , 0.96958222, 0.80305326],
               [ 0.83798987, 0.97324073, 0.80884306],
               [ 0.84290226, 0.97691284, 0.81464957]]
    for c in cm_data:
        c.append(1.0)
    cols=cols+cm_data
    precip_img = ax.pcolorfast(lons, lats, precip.data,
                               cmap=matplotlib.colors.ListedColormap(cols),
                               vmin=0,
                               vmax=1,
                               alpha=0.8,
                               zorder=200)


 
# Load the validation data
prate=twcr.load('prate',datetime.datetime(2010,3,12,18),
                           version='2c')
prate=tensor_cube(prate.extract(iris.Constraint(member=1)))
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
prate_t = tf.convert_to_tensor(normalise_precip(prate).data,numpy.float32)
prate_t = tf.reshape(prate_t,[79,159,1])
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
ict = tf.concat([t2m_t,prate_t,u10m_t,v10m_t],2) # Now [79,159,4]
ict = tf.reshape(ict,[1,79,159,4])
result = autoencoder.predict_on_batch(ict)
result = tf.reshape(result,[79,159,4])

# Convert the encoded fields back to unnormalised cubes 
t2m_r=t2m.copy()
t2m_r.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
t2m_r = unnormalise_t2m(t2m_r)
#t2m_r = unnormalise_t2m(normalise_t2m(t2m))
prate_r=prate.copy()
prate_r.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
prate_r = unnormalise_precip(prate_r)
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
three_plot(ax_original,t2m,u10m,v10m,prate)
ax_reconstructed=fig.add_axes([0.005,0.025,0.75,0.45])
three_plot(ax_reconstructed,t2m_r,u10m_r,v10m_r,prate_r)

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
ax_prate=fig.add_axes([0.83,0.55,0.16,0.17])
plot_scatter(ax_prate,prate.data,prate_r.data)
ax_u10m=fig.add_axes([0.83,0.30,0.16,0.17])
plot_scatter(ax_u10m,u10m.data,u10m_r.data)
ax_v10m=fig.add_axes([0.83,0.05,0.16,0.17])
plot_scatter(ax_v10m,v10m.data,v10m_r.data)

# Render the figure as a png
fig.savefig("comparison_results.png")


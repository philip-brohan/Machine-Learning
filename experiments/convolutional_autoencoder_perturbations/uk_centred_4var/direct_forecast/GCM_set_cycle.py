#!/usr/bin/env python

# Run the direct-forecast predictor for a year, forcing it with
#  insolation, and storing the real and latent space predictors 
#  every 6 hours.

import tensorflow as tf
tf.enable_eager_execution()
import numpy
import os
import pickle
import datetime
import math

import iris
import IRData.twcr as twcr

# Start on Jan 1st, 1989
dtstart=datetime.datetime(1989,3,1,0)

# Predictor model epoch
epoch=10

# Initialisation - load the data, regrid and normalise
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

def normalise_t2m(p):
   res=p.copy()
   res.data -= 280
   res.data /= 50
   return res

def normalise_wind(p):
   res=p.copy()
   res.data /= 12
   return res

def normalise_prmsl(p):
   res=p.copy()
   res.data -= 101325
   res.data /= 3000
   return res

def normalise_insolation(p):
   res=p.copy()
   res.data /= 25
   return res

# load the insolation data
def load_insolation(year,month,day,hour):
    if month==2 and day==29: day=28
    time_constraint=iris.Constraint(time=iris.time.PartialDateTime(
                                    year=1969,
                                    month=month,
                                    day=day,
                                    hour=hour))
    ic=iris.load_cube("%s/20CR/version_2c/ensmean/cduvb.1969.nc" % os.getenv('DATADIR'),
                      iris.Constraint(name='3-hourly Clear Sky UV-B Downward Solar Flux') &
                      time_constraint)
    coord_s=iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    ic.coord('latitude').coord_system=coord_s
    ic.coord('longitude').coord_system=coord_s
    return ic


# Load the starting data and reshape into a state vector tensor
prmsl = twcr.load('prmsl',dtstart,version='2c')
prmsl = tensor_cube(prmsl.extract(iris.Constraint(member=1)))
prmsl = tf.convert_to_tensor(normalise_prmsl(prmsl).data,numpy.float32)
prmsl = tf.reshape(prmsl,[79,159,1])
t2m = twcr.load('air.2m',dtstart,version='2c')
t2m = tensor_cube(t2m.extract(iris.Constraint(member=1)))
t2m = tf.convert_to_tensor(normalise_t2m(t2m).data,numpy.float32)
t2m = tf.reshape(t2m,[79,159,1])
u10m = twcr.load('uwnd.10m',dtstart,version='2c')
u10m = tensor_cube(u10m.extract(iris.Constraint(member=1)))
u10m = tf.convert_to_tensor(normalise_wind(u10m).data,numpy.float32)
u10m = tf.reshape(u10m,[79,159,1])
v10m = twcr.load('vwnd.10m',dtstart,version='2c')
v10m = tensor_cube(v10m.extract(iris.Constraint(member=1)))
v10m = tf.convert_to_tensor(normalise_wind(v10m).data,numpy.float32)
v10m = tf.reshape(v10m,[79,159,1])
insol = tensor_cube(load_insolation(dtstart.year,dtstart.month,dtstart.day,dtstart.hour))
insol = tf.convert_to_tensor(normalise_insolation(insol).data,numpy.float32)
insol = tf.reshape(insol,[79,159,1])

state_v = tf.concat([t2m,prmsl,u10m,v10m,insol],2) # Now [79,159,5]
state_v = tf.reshape(state_v,[1,79,159,5])

# Load the predictor
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_uk_centred_direct_forecast/"+
                  "saved_models/Epoch_%04d/predictor") % (
                      os.getenv('SCRATCH'),epoch)
predictor=tf.keras.models.load_model(model_save_file,compile=False)
# Also load the encoder (to get the latent state)
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_uk_centred_direct_forecast/"+
                  "saved_models/Epoch_%04d/encoder") % (
                      os.getenv('SCRATCH'),epoch)
encoder=tf.keras.models.load_model(model_save_file,compile=False)

# We want to constrain the forecasts so they follow the diurnal and annual cycles.
# Do this by generating multiple forecasts, and only keeping those with the
# correct cycle location. This requires a model to judge cycle location from
# the weather fields:
model_save_file=("%s/Machine-Learning-experiments/"+
                   "convolutional_autoencoder_perturbations/"+
                   "check_cycles/saved_models/Epoch_0024/encoder") % (
                         os.getenv('SCRATCH')) 
cycler=tf.keras.models.load_model(model_save_file,compile=False)


# Run forward in 6-hour increments
mdays=[0,31,59,90,120,151,181,212,243,273,304,334]
current=dtstart
while current<dtstart+datetime.timedelta(days=365):
    current += datetime.timedelta(hours=6)
    diurnal = current.hour/24
    dy = current.day
    if current.month==2 and current.day==29: dy=28
    annual=math.sin((3.141592*(mdays[current.month-1]+dy)/365))
    count=0
    while count<100:
        latent_s = encoder.predict_on_batch(state_v)
        new_v = predictor.predict_on_batch(state_v)
        cycle_i = tf.convert_to_tensor(new_v,numpy.float32)
        cycle_i = tf.reshape(cycle_i,[1,79,159,5])
        e_cycles = cycler.predict_on_batch(cycle_i[:,:,:,0:4])
        if (abs(annual-e_cycles[0,0])<0.05 and
            abs(diurnal-e_cycles[0,1])<1.0):
            state_v = new_v
            break
        count=count+1
        print(annual)
        print(e_cycles[0,0])
        print(diurnal)
        print(e_cycles[0,1])
        if count>99:
            raise Exception("Cycle match failure")

    pfile=("%s/Machine-Learning-experiments/GCM_mucdf_set_cycle/"+
           "%04d-%02d-%02d:%02d.pkl") % (os.getenv('SCRATCH'),
            current.year,current.month,current.day,current.hour)
    if not os.path.isdir(os.path.dirname(pfile)):
        os.makedirs(os.path.dirname(pfile))
    
    pickle.dump({'latent_s':latent_s,
                 'state_v':state_v},
                open(pfile,'wb'))

    # Replace calculated insolation with actual for the next step
    insol = tensor_cube(load_insolation(current.year,current.month,current.day,current.hour))
    insol = tf.convert_to_tensor(normalise_insolation(insol).data,numpy.float32)
    insol = tf.reshape(insol,[1,79,159,1])
    state_v = tf.concat([state_v[:,:,:,0:4],insol],3)
    state_v = tf.reshape(state_v,[1,79,159,5])

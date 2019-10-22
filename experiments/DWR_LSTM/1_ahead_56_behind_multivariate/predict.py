#!/usr/bin/env python

# Build an LSTM model to predict the next pressure at a station
#  using the 56 previous pressures (14 days worth).

import os
import pickle
import pandas
import numpy
import tensorflow as tf


# Model metadata - needed by validation and analysis scripts
model_meta={
    'station':'LONDON', # Target to reporduce
    'sources':['SCILLY', 'DUNGENESS', 'LONDON', 'VALENCIA', 'YARMOUTH', 'HOLYHEAD',  
         'BLACKSODPOINT', 'DONAGHADEE', 'SHIELDS', 'FORTWILLIAM', 'ABERDEEN', 
         'STORNOWAY', 'WICK'],
#    'sources':['LONDON'],     # List of stations to use as model input
    'random':False,   # Use a random series as a predictor
    'annual':True,    # Use the annual cycle as a predictor
    'diurnal':True,   # Use the diurnal cycle as a predictor
    'source_len':12,  # No of data points in the past to use
    'target_len':12,  # No of steps into the future to make prediction
    'persistence_error':{'prmsl':None,'air.2m':None}
}

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 10

# Model specification
# One layer, how many nodes?
lstm_node_n=128

# Get the position in the annual cycle
def get_annual(dates,year):
    dt=dates-numpy.datetime64('%04d-01-01' % year)
    days=dt.values/numpy.timedelta64(1,'D')
    annual=days/365
    return annual

# Estimate the persistence error
def get_persistence(obs):
    pd=(obs[model_meta[station]].values[model_meta[target_len]:]-
        obs[model_meta[station]].values[:(model_meta[target_len]*-1)])
    return numpy.std(pd)

def load_obs(variable,start_year,end_year):
    obs=None
    for year in range(start_year,end_year+1):
        of_name=(("%s/Machine-Learning-experiments/datasets/"+
                 "DWR/20CRv2c/%s/%04d.pkl") %
                           (os.getenv('SCRATCH'),variable,year))
        if not os.path.isfile(of_name):
            raise IOError("No obs file for given version and date")
        obs_f = pickle.load( open( of_name, "rb" ) )
        if model_meta['random']:
            obs_f['random']=numpy.random.rand(len(obs_f['Date']))
        if model_meta['annual']:
            obs_f['annual']=get_annual(obs_f['Date'],year)
        if model_meta['diurnal']:
            obs_f['diurnal']=obs_f['Date'].apply(lambda x: x.hour/24)
        if obs is None:
            obs=obs_f
        else:
            obs=pandas.concat([obs,obs_f])
    return obs

# Normalise the obs
def normalise(obs,variable):
    if variable == 'prmsl':
        res=obs.copy()
        res -= 101325
        res /= 3000
        return res
    if variable == 'air.2m':
        res=obs.copy()
        res -= 280
        res /= 50
        return res

# Make the target array from the obs
def make_target(prmsl,t2m):
    offset=model_meta['source_len']+model_meta['target_len']
    lgth=len(prmsl[model_meta['station']])-model_meta['target_len']-model_meta['source_len']
    target=numpy.empty((lgth,2))
    target[:,0]=normalise(prmsl[model_meta['station']].values[offset:],'prmsl')
    target[:,1]=normalise(t2m[model_meta['station']].values[offset:],'air.2m')
    return target

# Make the source array from the obs
def make_source(prmsl,t2m):
    offset=model_meta['source_len']
    lgth=len(prmsl[model_meta['station']])-model_meta['target_len']-model_meta['source_len']
    sdim=len(model_meta['sources'])*2
    for extra in ('random','annual','diurnal'):
        if model_meta[extra]: sdim += 1
    source=numpy.empty((lgth,sdim,model_meta['source_len']))
    for i in range(offset, len(prmsl[model_meta['station']])-model_meta['target_len']):
        idx=0
        for extra in ('random','annual','diurnal'):
            if model_meta[extra]:
                source[i-offset,idx,:]=prmsl[extra].values[i-offset:i]
                idx += 1
        for station in model_meta['sources']:
            source[i-offset,idx,:]=normalise(prmsl[station].values[i-offset:i],'prmsl')
            idx += 1
            source[i-offset,idx,:]=normalise(t2m[station].values[i-offset:i],'air.2m')
            idx += 1
    return source

prmsl=load_obs('prmsl',1969,2006)
t2m=load_obs('air.2m',1969,2006)
train_source=make_source(prmsl,t2m)
train_target=make_target(prmsl,t2m)
train_ds = tf.data.Dataset.from_tensor_slices((train_source, train_target))
train_ds=train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

prmsl=load_obs('prmsl',2006,2009)
t2m=load_obs('air.2m',2006,2009)
val_source=make_source(prmsl,t2m)
val_target=make_target(prmsl,t2m)
val_ds = tf.data.Dataset.from_tensor_slices((val_source, val_target))
val_ds=val_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Define an LSTM model
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(lstm_node_n, input_shape=train_source.shape[-2:]),
    tf.keras.layers.Dense(2)
])
simple_lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
simple_lstm_model.fit(train_ds,
                      epochs=EPOCHS,
                      steps_per_epoch=1000,
                      validation_data=val_ds, 
                      validation_steps=50)

# Save the model
save_file=("%s/Machine-Learning-experiments/"+
           "DWR_LSTM/1_ahead_56_behind_multivariate/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),EPOCHS)
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(simple_lstm_model,save_file)
# Save the training and normalisation parameters
meta_file="%s/meta_%04d" % (os.path.dirname(save_file),EPOCHS)
pickle.dump(model_meta, open(meta_file, "wb"))

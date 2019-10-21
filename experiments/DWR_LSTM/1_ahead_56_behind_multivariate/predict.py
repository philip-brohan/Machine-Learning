#!/usr/bin/env python

# Build an LSTM model to predict the next pressure at a station
#  using the 56 previous pressures (14 days worth).

import os
import pickle
import pandas
import numpy
import tensorflow as tf

# Target station
station='LONDON'

# Source stations
sources=['SCILLY', 'DUNGENESS', 'LONDON', 'VALENCIA', 'YARMOUTH', 'HOLYHEAD',  
         'BLACKSODPOINT', 'DONAGHADEE', 'SHIELDS', 'FORTWILLIAM', 'ABERDEEN', 
         'STORNOWAY', 'WICK']

# Source and target windows
source_len = 56
target_len = 4

# Normalisation parameters
s_mean = {}
s_std  = {}

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 10

# Model specification
# One layer, how many nodes?
lstm_node_n=32

# Get the position in the annual cycle
def get_annual(dates,year):
    dt=dates-numpy.datetime64('%04d-01-01' % year)
    days=dt.values/numpy.timedelta64(1,'D')
    annual=days/365
    return annual

# Load some data and arrange it for model fitting
def load_obs(variable,start_year,end_year):
    obs=None
    for year in range(start_year,end_year+1):
        of_name=(("%s/Machine-Learning-experiments/datasets/"+
                 "DWR/20CRv2c/%s/%04d.pkl") %
                           (os.getenv('SCRATCH'),variable,year))
        if not os.path.isfile(of_name):
            raise IOError("No obs file for given version and date")
        obs_f = pickle.load( open( of_name, "rb" ) )
        obs_f['annual']=get_annual(obs_f['Date'],year)
        obs_f['diurnal']=obs_f['Date'].apply(lambda x: x.hour/24)
        obs_f['random']=numpy.random.rand(len(obs_f['Date']))
        if obs is None:
            obs=obs_f
        else:
            obs=pandas.concat([obs,obs_f])
    
    # Estimate the persistence error
    pd=obs[station].values[target_len:]-obs[station].values[:(target_len*-1)]
    global persistence_error
    persistence_error=numpy.std(pd)

    # Normalise the station series
    global s_mean
    s_mean[variable] = numpy.mean(obs[sources].values)
    global s_std
    s_std[variable]  = numpy.std(obs[sources].values)
    obs[sources] -= s_mean[variable]
    obs[sources] /= s_std[variable]

    # Batch up the station series into source & target chunks
    source = []
    target = []
    indices = ['random','annual','diurnal']
    indices.extend(sources)
    #indices=['random','annual','diurnal']
    for i in range(source_len, len(obs[station])-target_len):
        source.append(numpy.reshape(obs[indices].values[i-source_len:i], 
                                                (len(indices), source_len)))
        target.append(obs[station].values[i+target_len-1])
    source=numpy.array(source)
    target=numpy.array(target)
    return (source,target)

# Get both prmsl and t2m in the same array
def get_both(start_year,end_year):
    prmsl=load_obs('prmsl',start_year,end_year)
    t2m=load_obs('air.2m',start_year,end_year)
    obs=(numpy.concatenate((prmsl[0],t2m[0][:,3:,:]),axis=1),
         numpy.stack((prmsl[1],t2m[1]),axis=1))
    return obs
    
# Get two batches of data, one for training, the other for validation
(train_source,train_target)=get_both(1969,2005)
train_ds = tf.data.Dataset.from_tensor_slices((train_source, train_target))
train_ds=train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
(val_source,val_target)=get_both(2006,2009)
val_ds = tf.data.Dataset.from_tensor_slices((val_source, val_target))
val_ds=val_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Define an LSTM model
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(lstm_node_n, input_shape=train_source.shape[-2:]),
    tf.keras.layers.Dense(2)
])
simple_lstm_model.compile(optimizer='adam', loss='mae')

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
model_meta={'s_mean':s_mean,
            's_std':s_std,
            'persistence_error':persistence_error,
            'source_len':source_len,
            'target_len':target_len}
meta_file="%s/meta_%04d" % (os.path.dirname(save_file),EPOCHS)
pickle.dump(model_meta, open(meta_file, "wb"))

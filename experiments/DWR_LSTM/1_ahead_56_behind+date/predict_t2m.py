#!/usr/bin/env python

# Build an LSTM model to predict the next pressure at a station
#  using the 56 previous pressures (14 days worth).

import os
import pickle
import pandas
import numpy
import tensorflow as tf

# Work on only one station
station='LONDON'

# Source and target windows
source_len = 56
target_len = 12

# Normalisation parameters
s_mean = None
s_std  = None

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 10

# Model specification
# One layer, how many nodes?
lstm_node_n=32

# Get the position in the anual cycle
def get_annual(dates,year):
    dt=dates-numpy.datetime64('%04d-01-01' % year)
    days=dt.values/numpy.timedelta64(1,'D')
    annual=days/365
    return annual

# Load some data and arrange it for model fitting
def load_obs(start_year,end_year):
    obs=None
    for year in range(start_year,end_year+1):
        of_name=(("%s/Machine-Learning-experiments/datasets/"+
                 "DWR/20CRv2c/air.2m/%04d.pkl") %
                           (os.getenv('SCRATCH'),year))
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
    if s_mean is None:
        s_mean = numpy.mean(obs[station])
    global s_std
    if s_std is None:
        s_std  = numpy.std(obs[station])
    obs[station] -= s_mean
    obs[station] /= s_std

    # Batch up the station series into source & target chunks
    source = []
    target = []
    indices = ['random','annual','diurnal',station]
    #indices=['random','annual','diurnal']
    for i in range(source_len, len(obs[station])-target_len):
        source.append(numpy.reshape(obs[indices].values[i-source_len:i], 
                                                (len(indices), source_len)))
        target.append(obs[station].values[i+target_len-1])
    source=numpy.array(source)
    target=numpy.array(target)
    return (source,target)
    

# Get two batches of data, one for training, the other for validation
(train_source,train_target)=load_obs(1969,2005)
train_ds = tf.data.Dataset.from_tensor_slices((train_source, train_target))
train_ds=train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
(val_source,val_target)=load_obs(2006,2009)
val_ds = tf.data.Dataset.from_tensor_slices((val_source, val_target))
val_ds=val_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Define an LSTM model
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(lstm_node_n, input_shape=train_source.shape[-2:]),
    tf.keras.layers.Dense(1)
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
           "DWR_LSTM/1_ahead_56_behind+date/air.2m/"+
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

#!/usr/bin/env python

# Build an LSTM model to forecast the time-evolution of the latent
#  space weather fields


import os
import pickle
import numpy
import datetime
import tensorflow as tf
tf.enable_eager_execution()

# Set the model parameters from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--extras", help="Extra predictors",
                    type=str,default=None,action='append')
parser.add_argument("--source_len", help="No of previous steps to use",
                    type=int,required=False,default=4)
parser.add_argument("--target_len", help="Steps forward to predict",
                    type=int,required=False,default=1)
parser.add_argument("--epochs", help="Epochs to run",
                    type=int,required=False,default=10)
parser.add_argument("--n_nodes", help="No. of LSTM nodes",
                    type=int,required=False,default=100)
parser.add_argument("--opdir", help="Directory for output files",
                    type=str,required=False,default='default')
args = parser.parse_args()
args.opdir=("%s/Machine-Learning-experiments/LS_LSTM/%s" %
               (os.getenv('SCRATCH'),args.opdir))
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

model_meta={}
model_meta['random']=False
model_meta['annual']=False
model_meta['diurnal']=False
if args.extras is not None:
    for extra in args.extras:
        if extra == 'random': 
            model_meta['random']=True
        elif extra == 'annual': 
            model_meta['annual']=True
        elif extra == 'diurnal': 
            model_meta['diurnal']=True
        else:
            raise Exception('Unsupported extra')
model_meta['source_len']=args.source_len
model_meta['target_len']=args.target_len

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = args.epochs

# Model specification
# One layer, how many nodes?
lstm_node_n=args.n_nodes

# Get the position in the annual cycle
def get_annual(dates,year):
    dt=dates-numpy.datetime64('%04d-01-01' % year)
    days=dt.values/numpy.timedelta64(1,'D')
    annual=days/365
    return annual

def load_ls(start,end):
    Dates=[]
    current=start
    while current <= end:
        Dates.append(current)
        current += datetime.timedelta(hours=6)
    Dates=numpy.array(Dates)
    ls=numpy.empty((len(Dates),100))
    if model_meta['random']:  random  = numpy.random.rand(len(Dates))
    if model_meta['annual']:  annual  = numpy.empty((len(Dates)))
    if model_meta['diurnal']: diurnal = numpy.empty((len(Dates)))
    for idx,Date in enumerate(Dates):
        l_file=(("%s/Machine-Learning-experiments/datasets/latent_space/"+
                  "%04d-%02d-%02d:%02d.tfd") %
                       (os.getenv('SCRATCH'),
                        Date.year,Date.month,Date.day,Date.hour))
        sict  = tf.read_file(l_file)
        ls[idx,:]=tf.parse_tensor(sict,numpy.float32).numpy()
        if model_meta['annual']:
            annual[idx] = Date.timetuple().tm_yday
        if model_meta['diurnal']:
            diurnal[idx] = Date.hour/24
    result={'Date':Dates,'ls':ls}
    if model_meta['random']:  result['random']  = random
    if model_meta['annual']:  result['annual']  = annual
    if model_meta['diurnal']: result['diurnal'] = diurnal
    return result

# Make the target array from the obs
def make_target(obs):
    offset=model_meta['source_len']+model_meta['target_len']
    target=obs['ls'][offset:,:]
    return target

# Make the source array from the obs
def make_source(obs):
    offset=model_meta['source_len']
    lgth=len(obs['Date'])-model_meta['target_len']-model_meta['source_len']
    sdim=100
    for extra in ('random','annual','diurnal'):
        if model_meta[extra]: sdim += 1
    source=numpy.empty((lgth,sdim,model_meta['source_len']))
    for i in range(offset, len(obs['Date'])-model_meta['target_len']):
        idx=0
        for extra in ('random','annual','diurnal'):
            if model_meta[extra]:
                source[i-offset,idx,:]=obs[extra][i-offset:i]
                idx += 1
        source[i-offset,idx:,:]=numpy.transpose(obs['ls'][i-offset:i])
    return source

obs=load_ls(datetime.datetime(1969,1,1,0),datetime.datetime(1973,12,31,18))
train_source=make_source(obs)
train_target=make_target(obs)
train_ds = tf.data.Dataset.from_tensor_slices((train_source, train_target))
train_ds=train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

obs=load_ls(datetime.datetime(1974,1,1,0),datetime.datetime(1974,12,31,18))
val_source=make_source(obs)
val_target=make_target(obs)
val_ds = tf.data.Dataset.from_tensor_slices((val_source, val_target))
val_ds=val_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Define an LSTM model
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(lstm_node_n, input_shape=train_source.shape[-2:]),
    tf.keras.layers.Dense(100)
])
simple_lstm_model.compile(optimizer='adadelta', loss='mean_squared_error')

# Train the model
simple_lstm_model.fit(train_ds,
                      epochs=EPOCHS,
                      steps_per_epoch=1000,
                      validation_data=val_ds, 
                      validation_steps=500)

# Save the model
save_file="%s/predictor" % args.opdir
if not os.path.isdir(os.path.dirname(save_file)):
    os.makedirs(os.path.dirname(save_file))
tf.keras.models.save_model(simple_lstm_model,save_file)
# Save the training and normalisation parameters
meta_file="%s/meta.pkl" % args.opdir
pickle.dump(model_meta, open(meta_file, "wb"))

#!/usr/bin/env python

# Plot the pseudo-DWR obs for one station for a selected period

import os
import pickle
import datetime
import pandas
import numpy
import tensorflow as tf

import matplotlib
from matplotlib.backends.backend_agg import \
                 FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="Start time: YYYY-MM-DD:HH",
                    type=str,required=True)
parser.add_argument("--end", help="End time: YYYY-MM-DD:HH",
                    type=str,required=True)
parser.add_argument("--station", help="Station series to show",
                    type=str,required=False,default='LONDON')
parser.add_argument("--epoch", help="Use saved model at which epoch",
                    type=int,required=True)
args = parser.parse_args()
args.start=datetime.datetime.strptime(args.start,"%Y-%m-%d:%H")
args.end  =datetime.datetime.strptime(args.end,  "%Y-%m-%d:%H")

# Load the data
obs=None
year=args.start.year
while year<=args.end.year:
    of_name=(("%s/Machine-Learning-experiments/datasets/"+
             "DWR/20CRv2c/prmsl/%04d.pkl") %
                       (os.getenv('SCRATCH'),year))
    if not os.path.isfile(of_name):
        raise IOError("No obs file for given version and date")
    obs_f = pickle.load( open( of_name, "rb" ) )
    obs_f = obs_f[(obs_f['Date']>=args.start) & (obs_f['Date']<args.end)]
    if obs is None:
        obs=obs_f
    else:
        obs=pandas.concat([obs,obs_f])
    year += 1

# Load the saved model
save_file=("%s/Machine-Learning-experiments/"+
           "DWR_LSTM/1_ahead_56_behind/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),args.epoch)
predictor=tf.keras.models.load_model(save_file)
# Retrieve the model metadata
meta_file="%s/meta_%04d" % (os.path.dirname(save_file),args.epoch)
model_meta=pickle.load( open( meta_file, "rb" ) )

# For each data point in the selected period, calculate a prediction
obs[args.station] -= model_meta['s_mean']
obs[args.station] /= model_meta['s_std']
prediction=[]
for i in range(model_meta['source_len'], 
               len(obs[args.station])-model_meta['target_len']):
    source=numpy.reshape(obs[args.station].values[i-model_meta['source_len']:i], 
                                            (1, model_meta['source_len'], 1))
    prediction.append(predictor.predict(source)[0])
prediction=numpy.array(prediction).flatten()

# Unnormalise the data
target=obs[args.station].values*model_meta['s_std']+model_meta['s_mean']
prediction = prediction*model_meta['s_std']+model_meta['s_mean']

# Make the plot
aspect=16.0/9.0
fig=Figure(figsize=(10.8*aspect,10.8),  # Width, Height (inches)
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)
font = {'family' : 'sans-serif',
        'sans-serif' : 'Arial',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

# Single axes - var v. time
ax=fig.add_axes([0.05,0.05,0.945,0.74])
ax.set_xlim(min(obs['Date'].values),
            max(obs['Date'].values))
ax.set_ylim(min(min(prediction),min(target)-model_meta['persistence_error']*2)/100-1,
            max(max(prediction),max(target)+model_meta['persistence_error']*2)/100+1)
#per_ep=numpy.array([numpy.concatenate((obs['Date'].values,numpy.flip(obs['Date'].values))),
#                    numpy.concatenate(((target+model_meta['persistence_error']*2)/100,
#                               (target-model_meta['persistence_error']*2)/100))])
#ax.add_patch(Polygon(numpy.transpose(per_ep),color='grey',zorder=10))
ax.plot(obs['Date'].values,
                (target+model_meta['persistence_error']*2)/100,
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax.plot(obs['Date'].values,
                (target-model_meta['persistence_error']*2)/100,
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)

ax.plot(obs['Date'].values,
                target/100,
                markersize=10,
                color='black',
                marker='.',
                linewidth=1.0,
                alpha=1.0,
                zorder=50)
ax.plot(obs['Date'].values[model_meta['source_len']:(len(obs[args.station])-model_meta['target_len'])],
                prediction/100,
                markersize=10,
                color='blue',
                marker='.',
                linewidth=1.0,
                alpha=1.0,
                zorder=60)
# Also plot the error - compared with persistence
persistence = target[model_meta['target_len']:]-target[:(model_meta['target_len']*-1)]
delta_p=persistence
delta_m=target[model_meta['source_len']:(len(obs[args.station])-model_meta['target_len'])]-prediction
ax2=fig.add_axes([0.05,0.80,0.945,0.195])
ax2.set_xlim(min(obs['Date'].values),
             max(obs['Date'].values))
ax2.get_xaxis().set_visible(False)
ax2.set_ylim(min(min(delta_p),min(delta_m))/100-2,
             max(max(delta_p),max(delta_m))/100+2)
ax2.set_ylabel('MSLP (hPa)')
ax2.plot(numpy.array([min(obs['Date'].values),max(obs['Date'].values)]),
         numpy.array([0,0]),
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax2.plot(obs['Date'].values[model_meta['target_len']:],
                delta_p/100,
                markersize=6,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
ax2.plot(obs['Date'].values[model_meta['source_len']:(len(obs[args.station])-model_meta['target_len'])],
                delta_m/100,
                markersize=5,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=60)


fig.savefig('Predictions.png')

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
parser.add_argument("--epoch", help="Use saved model at which epoch",
                    type=int,required=True)
args = parser.parse_args()
args.start=datetime.datetime.strptime(args.start,"%Y-%m-%d:%H")
args.end  =datetime.datetime.strptime(args.end,  "%Y-%m-%d:%H")


# Load the saved model
save_file=("%s/Machine-Learning-experiments/"+
           "DWR_LSTM/1_ahead_56_behind_multivariate/"+
           "saved_models/Epoch_%04d") % (
                 os.getenv('SCRATCH'),args.epoch)
predictor=tf.keras.models.load_model(save_file)
# Retrieve the model metadata
meta_file="%s/meta_%04d" % (os.path.dirname(save_file),args.epoch)
model_meta=pickle.load( open( meta_file, "rb" ) )


# Get the position in the annual cycle
def get_annual(dates,year):
    dt=dates-numpy.datetime64('%04d-01-01' % year)
    days=dt.values/numpy.timedelta64(1,'D')
    annual=days/365
    return annual

# Estimate the persistence error
def get_persistence(obs):
    pd=(obs[model_meta['station']].values[model_meta['target_len']:]-
        obs[model_meta['station']].values[:(model_meta['target_len']*-1)])
    return pd

# Load the data
def load_var(variable):
    obs=None
    year=args.start.year
    while year<=args.end.year:
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
        obs_f = obs_f[(obs_f['Date']>=args.start) & (obs_f['Date']<args.end)]
        if obs is None:
            obs=obs_f
        else:
            obs=pandas.concat([obs,obs_f])
        year += 1
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
def unnormalise(obs,variable):
    if variable == 'prmsl':
        res=obs.copy()
        res *= 3000
        res += 101325
        return res
    if variable == 'air.2m':
        res=obs.copy()
        res *= 50
        res += 280
        return res

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

prmsl=load_var('prmsl')
t2m=load_var('air.2m')
source=make_source(prmsl,t2m)

# For each data point in the selected period, calculate a prediction
prediction={'prmsl':[],'t2m':[]}
p=predictor.predict(source)
prediction['prmsl']=unnormalise(p[:,0],'prmsl')
prediction['air.2m']=unnormalise(p[:,1],'air.2m')

# Identify the target series
target={'prmsl':prmsl[model_meta['station']],
        'air.2m':t2m[model_meta['station']]}

# Estimate the persistence errors
persistence={'prmsl':get_persistence(prmsl),
             'air.2m':get_persistence(t2m)}
persistence_error={'prmsl':numpy.std(persistence['prmsl']),
                   'air.2m':numpy.std(persistence['air.2m'])}

# Get the date series for the predictions
pDates=prmsl['Date'].values[(model_meta['source_len']+model_meta['target_len']):]

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

# Bottom axes - pressure v. time
ax=fig.add_axes([0.05,0.05,0.945,0.28])
ax.set_xlim(min(prmsl['Date'].values),
            max(prmsl['Date'].values))
ax.set_ylim(min(min(prediction['prmsl']),
                min(target['prmsl'])-persistence_error['prmsl']*2)/100-1,
                max(max(prediction['prmsl']),
                max(target['prmsl'])+persistence_error['prmsl']*2)/100+1)
ax.set_ylabel('MSLP (hPa)')
ax.grid()
ax.plot(prmsl['Date'].values,
                (target['prmsl']+persistence_error['prmsl']*2)/100,
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax.plot(prmsl['Date'].values,
                (target['prmsl']-persistence_error['prmsl']*2)/100,
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax.plot(prmsl['Date'].values,
                target['prmsl']/100,
                markersize=6,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
ax.plot(pDates,
                prediction['prmsl']/100,
                markersize=6,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=60)
# Next axes - t2m v. time
ax2=fig.add_axes([0.05,0.35,0.945,0.28])
ax2.set_xlim(min(t2m['Date'].values),
            max(t2m['Date'].values))
ax2.set_ylim(min(min(prediction['air.2m']),
                 min(target['air.2m'])-persistence_error['air.2m']*2)-273.15-1,
                 max(max(prediction['air.2m']),
                 max(target['air.2m'])+persistence_error['air.2m']*2)-273.15+1)
for ticm in ax2.xaxis.get_major_ticks():
    ticm.label.set_fontsize(0)
ax2.set_ylabel('T2m (C)')
ax2.grid()
ax2.plot(prmsl['Date'].values,
                (target['air.2m']+persistence_error['air.2m']*2)-273.15,
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax2.plot(prmsl['Date'].values,
                (target['air.2m']-persistence_error['air.2m']*2)-273.15,
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax2.plot(t2m['Date'].values,
                target['air.2m']-273.15,
                markersize=6,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
ax2.plot(pDates,
                prediction['air.2m']-273.15,
                markersize=6,
                color='red',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=60)

# pressure error - compared with persistence
delta_p=persistence['prmsl']
delta_m=(target['prmsl'][model_meta['source_len']:(len(target['prmsl'])-model_meta['target_len'])]-
         prediction['prmsl'])
ax3=fig.add_axes([0.05,0.66,0.945,0.145])
ax3.set_xlim(min(prmsl['Date'].values),
             max(prmsl['Date'].values))
ax3.set_ylim(min(min(delta_p),min(delta_m))/100-2,
             max(max(delta_p),max(delta_m))/100+2)
ax3.set_ylabel('MSLP (hPa)')
for ticm in ax3.xaxis.get_major_ticks():
    ticm.label.set_fontsize(0)
ax2.set_ylabel('T2m (C)')
ax3.grid()
ax3.plot(numpy.array([min(prmsl['Date'].values),max(prmsl['Date'].values)]),
         numpy.array([0,0]),
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax3.plot(prmsl['Date'].values[model_meta['target_len']:],
                delta_p/100,
                markersize=6,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
ax3.plot(pDates,
                delta_m/100,
                markersize=5,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=60)

# temperature error - compared with persistence
delta_p=persistence['air.2m']
delta_m=(target['air.2m'][model_meta['source_len']:(len(target['air.2m'])-model_meta['target_len'])]-
         prediction['air.2m'])
ax4=fig.add_axes([0.05,0.81,0.945,0.145])
ax4.set_xlim(min(t2m['Date'].values),
             max(t2m['Date'].values))
ax4.set_ylim(min(min(delta_p),min(delta_m))-2,
             max(max(delta_p),max(delta_m))+2)
ax4.set_ylabel('T2m (C)')
for ticm in ax4.xaxis.get_major_ticks():
    ticm.label.set_fontsize(0)
ax4.grid()
ax4.plot(numpy.array([min(t2m['Date'].values),max(t2m['Date'].values)]),
         numpy.array([0,0]),
                markersize=0,
                color='grey',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax4.plot(t2m['Date'].values[model_meta['target_len']:],
                delta_p,
                markersize=6,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
ax4.plot(pDates,
                delta_m,
                markersize=5,
                color='red',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=60)

fig.savefig('Predictions.png')

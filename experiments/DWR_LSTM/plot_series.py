#!/usr/bin/env python

# Plot the pseudo-DWR obs for one station for a selected period

import os
import pickle
import datetime
import pandas

import matplotlib
from matplotlib.backends.backend_agg import \
                 FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="Start time: YYYY-MM-DD:HH",
                    type=str,required=True)
parser.add_argument("--end", help="End time: YYYY-MM-DD:HH",
                    type=str,required=True)
parser.add_argument("--station", help="Station series to show",
                    type=str,required=False,default='LONDON')
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
ax=fig.add_axes([0.05,0.05,0.945,0.94])
# Axes ranges
#ax.set_xlim(args.start,args.end)
ax.set_xlim(min(obs['Date'].values),
            max(obs['Date'].values))
ax.set_ylim(min(obs[args.station])/100-1,
            max(obs[args.station])/100+1)
ax.set_ylabel('MSLP (hPa)')
ax.plot(obs['Date'].values,
           obs[args.station].values/100,
                markersize=10,
                color='blue',
                marker='.',
                linewidth=1.0,
                alpha=1.0,
                zorder=50)

fig.savefig('Series.png')

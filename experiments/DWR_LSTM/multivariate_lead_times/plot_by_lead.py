#!/usr/bin/env python

# Plot the LSTM model results as a function of lead time

import os
import pickle
import numpy
import matplotlib
from matplotlib.backends.backend_agg import \
                 FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--opdir", help="Directory for output files",
                    type=str,required=False,default='default')
parser.add_argument("--cmpdir", help="Directory for comparison output",
                    type=str,required=False,default=None)
parser.add_argument('--annual', dest='annual', action='store_true')
parser.add_argument('--no-annual', dest='annual', action='store_false')
parser.set_defaults(annual=True)
args = parser.parse_args()
args.opdir=("%s/Machine-Learning/experiments/DWR_LSTM/multivariate_lead_times/%s" %
               (os.getenv('SCRATCH'),args.opdir))

persistence={'prmsl':[],'air.2m':[]}
model={'prmsl':[],'air.2m':[]}
# Load the model errors
for lead in range(1,41):
   res=pickle.load(open("%s/errors_%02d.pkl" % (args.opdir,lead), "rb"))
   for var in ('prmsl','air.2m'):
      persistence[var].append(res['persistence_error'][var])
      model[var].append(res['delta_m'][var])
for var in ('prmsl','air.2m'):
    persistence[var]=numpy.array(persistence[var])
    model[var]=numpy.array(model[var])

# If selected, load the annual cycle errors
annual={'prmsl':[],'air.2m':[]}
adir=("%s/Machine-Learning/experiments/DWR_LSTM/multivariate_lead_times/%s" %
               (os.getenv('SCRATCH'),'annual'))
if args.annual:
    for lead in range(1,41):
       res=pickle.load(open("%s/errors_%02d.pkl" % (adir,lead), "rb"))
       for var in ('prmsl','air.2m'):
          annual[var].append(res['delta_m'][var])
    for var in ('prmsl','air.2m'):
        annual[var]=numpy.array(annual[var])

# If selected, load the comparison errors
comparison={'prmsl':[],'air.2m':[]}
cdir=("%s/Machine-Learning/experiments/DWR_LSTM/multivariate_lead_times/%s" %
               (os.getenv('SCRATCH'),args.cmpdir))
if args.annual:
    for lead in range(1,41):
       res=pickle.load(open("%s/errors_%02d.pkl" % (cdir,lead), "rb"))
       for var in ('prmsl','air.2m'):
          comparison[var].append(res['delta_m'][var])
    for var in ('prmsl','air.2m'):
        comparison[var]=numpy.array(comparison[var])

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
        'size'   : 18}
matplotlib.rc('font', **font)

# Left axes - mslp error v lead
ax=fig.add_axes([0.05,0.07,0.425,0.9])
ax.set_xlim(0,10.25)
ax.set_xlabel('Lead time (days)')
ax.set_ylim(0,max(max(persistence['prmsl']),max(model['prmsl']))/100+1)
ax.set_ylabel('MSLP (hPa)')
ax.grid()
ax.plot(numpy.linspace(0.25,10,40),
                persistence['prmsl']/100,
                markersize=10,
                color='black',
                marker='.',
                linewidth=2,
                alpha=1.0,
                zorder=30)
if args.annual:
    ax.plot(numpy.linspace(0.25,10,40),
                    annual['prmsl']/100,
                    markersize=10,
                    color='grey',
                    marker='.',
                    linewidth=2,
                    alpha=1.0,
                    zorder=20)
if args.cmpdir is not None:
    ax.plot(numpy.linspace(0.25,10,40),
                    comparison['prmsl']/100,
                    markersize=10,
                    color='blue',
                    marker='.',
                    linewidth=2,
                    alpha=0.5,
                    zorder=25)

ax.plot(numpy.linspace(0.25,10,40),
                model['prmsl']/100,
                markersize=10,
                color='blue',
                marker='.',
                linewidth=2,
                alpha=1.0,
                zorder=30)
# Right axes -air.2m' error v lead
ax2=fig.add_axes([0.525,0.07,0.425,0.9])
ax2.set_xlim(0,10.25)
ax2.set_xlabel('Lead time (days)')
ax2.set_ylim(0,max(max(persistence['air.2m']),max(model['air.2m']))+1)
ax2.set_ylabel('T2m (C)')
ax2.grid()
ax2.plot(numpy.linspace(0.25,10,40),
                persistence['air.2m'],
                markersize=10,
                color='black',
                marker='.',
                linewidth=2,
                alpha=1.0,
                zorder=30)
if args.annual:
    ax2.plot(numpy.linspace(0.25,10,40),
                    annual['air.2m'],
                    markersize=10,
                    color='grey',
                    marker='.',
                    linewidth=2,
                    alpha=1.0,
                    zorder=20)
if args.cmpdir is not None:
    ax2.plot(numpy.linspace(0.25,10,40),
                    comparison['air.2m'],
                    markersize=10,
                    color='red',
                    marker='.',
                    linewidth=2,
                    alpha=0.5,
                    zorder=25)
ax2.plot(numpy.linspace(0.25,10,40),
                model['air.2m'],
                markersize=10,
                color='red',
                marker='.',
                linewidth=2,
                alpha=1.0,
                zorder=30)

fig.savefig('By_lead.png')

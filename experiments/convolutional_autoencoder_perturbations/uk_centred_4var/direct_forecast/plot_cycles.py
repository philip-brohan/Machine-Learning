#!/usr/bin/env python

# Use the annual and diurnal cycle inference model to check the ML GCM

import os
import pickle
import datetime
import numpy
import tensorflow as tf
tf.enable_eager_execution()
import IRData.twcr as twcr
import iris
import math

import matplotlib
from matplotlib.backends.backend_agg import \
                 FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", help="Start time: YYYY-MM-DD:HH",
                    type=str,default='1989-03-01:06',required=False)
parser.add_argument("--end", help="End time: YYYY-MM-DD:HH",
                    type=str,default='1989-06-30:18',required=False)
args = parser.parse_args()
args.start=datetime.datetime.strptime(args.start,"%Y-%m-%d:%H")
args.end  =datetime.datetime.strptime(args.end,  "%Y-%m-%d:%H")

# Load the cycle-check model
model_dir = (("%s/Machine-Learning-experiments/"+
             "convolutional_autoencoder_perturbations/"+
             "/check_cycles/saved_models/Epoch_0024") %
               (os.getenv('SCRATCH')))
save_file = "%s/encoder" % model_dir
cycler = tf.keras.models.load_model(save_file)


#  load the 20CR data and calculate the cycles
def get_cycles(dte):
            
    pfile=("%s/Machine-Learning-experiments/GCM_mucdf/"+
           "%04d-%02d-%02d:%02d.pkl") % (os.getenv('SCRATCH'),
            current.year,current.month,current.day,current.hour)

    state = pickle.load(open(pfile,'rb'))
    e_cycles = cycler.predict_on_batch(state['state_v'][:,:,:,0:4])
    return e_cycles 

# For each time in the validation period,
a_annual  = []
a_diurnal = []
e_annual  = []
e_diurnal = []
dates     = []
current=args.start
mdays=[0,31,59,90,120,151,181,212,243,273,304,334]
while current<args.end:
    dates.append(current)
    a_diurnal.append(current.hour/24)
    dy=current.day
    if current.month==2 and current.day==29: dy=28
    a_annual.append(math.sin((3.141592*(mdays[current.month-1]+dy)/365)))
    encoded = get_cycles(current)
    e_annual.append(encoded[0,0])
    e_diurnal.append(encoded[0,1])
    current += datetime.timedelta(hours=30)

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

# Bottom axes - annual v. time
ax=fig.add_axes([0.05,0.05,0.945,0.4])
ax.set_xlim(min(dates),
            max(dates))
ax.set_ylim(-0.1,1.1)
ax.set_ylabel('Annual Cycle')
ax.grid()
ax.plot(dates,a_annual,
                markersize=5,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax.plot(dates,e_annual,
                markersize=5,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)
# Next axes - t2m v. time
ax2=fig.add_axes([0.05,0.5,0.945,0.4])
ax2.set_xlim(min(dates),
            max(dates))
ax2.set_ylim(-0.1,0.8)
ax2.set_ylabel('Diurnal Cycle')
ax2.grid()
ax2.plot(dates,a_diurnal,
                markersize=5,
                color='black',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=30)
ax2.plot(dates,e_diurnal,
                markersize=5,
                color='blue',
                marker='.',
                linewidth=0.5,
                alpha=1.0,
                zorder=50)

fig.savefig('Cycles.png')

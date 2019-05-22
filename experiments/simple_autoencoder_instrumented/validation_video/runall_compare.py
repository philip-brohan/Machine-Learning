#!/usr/bin/env python

# Make all the individual frames for a movie

import os
import subprocess
import datetime

# Where to put the output files
opdir="%s/slurm_output" % os.getenv('SCRATCH')
if not os.path.isdir(opdir):
    os.makedirs(opdir)

# Function to check if the job is already done for this timepoint
def is_done(year,month,day,hour,epoch):
    op_file_name=("%s/Machine-Learning-experiments/"+
                   "simple_autoencoder_instrumented/"+
                   "images/%02d%02d%02d%02d_%04d.png") % (
                          os.getenv('SCRATCH'),
                          year,month,
                          day,hour,
                          epoch)
    if os.path.isfile(op_file_name):
        return True
    return False

f=open("run.txt","w+")

start_day=datetime.datetime(2009, 1, 1, 0)
end_day  =datetime.datetime(2009, 1,31,23)

current_day=start_day
while current_day<=end_day:
    if is_done(current_day.year,current_day.month,
                   current_day.day,current_day.hour,1000):
        current_day=current_day+datetime.timedelta(hours=1)
        continue
    cmd=("./compare.py --year=%d --month=%d" +
        " --day=%d --hour=%d --epoch=1000\n") % (
           current_day.year,current_day.month,
           current_day.day,current_day.hour)
    f.write(cmd)
    current_day=current_day+datetime.timedelta(hours=1)
f.close()


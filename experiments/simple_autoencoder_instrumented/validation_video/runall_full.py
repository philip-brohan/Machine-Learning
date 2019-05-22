#!/usr/bin/env python

# Make a comparison plot for each epoch 1-1000
# Actually make a list of commands to do that, 
#              which can then be run in parallel.


import os
import subprocess
import datetime

# Where to put the output files
opdir="%s/slurm_output" % os.getenv('SCRATCH')
if not os.path.isdir(opdir):
    os.makedirs(opdir)

# Function to check if the job is already done for this epoch
def is_done(epoch):
    op_file_name=("%s/Machine-Learning-experiments/"+
                   "simple_autoencoder_instrumented/"+
                   "images/comparison_%04d.png") % (
                          os.getenv('SCRATCH'),epoch)
    if os.path.isfile(op_file_name):
        return True
    return False

f=open("run.txt","w+")

epoch=1
while epoch<=1000:
    if is_done(epoch):
        epoch=epoch+1
        continue
    cmd="./compare_full.py --epoch=%d \n" % epoch 
    f.write(cmd)
    epoch=epoch+1
f.close()




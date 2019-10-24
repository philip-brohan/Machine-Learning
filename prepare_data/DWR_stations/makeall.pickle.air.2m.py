#!/usr/bin/env python

# Pickle fake t2m data from the DWR stations for each year 1969-2010

import os

# Function to check if the job is already done for this timepoint
def is_done(year):
    op_file_name=("%s/Machine-Learning-experiments/datasets/DWR/20CRv2c/air.2m/" +
                  "%04d.pkl") % (
                            os.getenv('SCRATCH'),year)
    if os.path.isfile(op_file_name):
        return True
    return False

f=open("run.txt","w+")

for year in range(1969,2011):

    if is_done(year): continue
    cmd=("./make_annual_pickle.py --year=%d --variable=air.2m \n" % year )
    f.write(cmd)
    
f.close()


#!/usr/bin/env python

# Pickle fake prmsl data from the DWR stations for each year 1969-2010

import os

# Function to check if the job is already done for this timepoint
def is_done(year):
    op_file_name=("%s/Machine-Learning-experiments/datasets/DWR/20CR2c/prmsl/" +
                  "%04d.pkl") % (
                            os.getenv('SCRATCH'),year)
    if os.path.isfile(op_file_name):
        return True
    return False

f=open("run.txt","w+")

for year in range(1969,2011):

    cmd=("./make_annual_pickle.py --year=%d \n" % year )
    f.write(cmd)
    
f.close()


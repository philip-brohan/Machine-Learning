#!/usr/bin/env python

# Make a list of the commands needed to make tf data files with 20CR
# data in the latent space

import os
import datetime

# Function to check if the job is already done for this timepoint
def is_done(year,month,day,hour):
    op_file_name=("%s/Machine-Learning-experiments/datasets/latent_space/"+
                  "%04d-%02d-%02d:%02d.tfd") % (
                            os.getenv('SCRATCH'),
                            year,month,day,hour)
    if os.path.isfile(op_file_name):
        return True
    return False

f=open("run.txt","w+")

start_day=datetime.datetime(1969,  1,  1,  0)
end_day  =datetime.datetime(2009, 12, 31, 18)

current_day=start_day
while current_day<=end_day:
    if not is_done(current_day.year,current_day.month,
                   current_day.day,current_day.hour):
        cmd=("./make_ls_tensor.py --year=%d --month=%d" +
            " --day=%d --hour=%d --epoch=10\n") % (
               current_day.year,current_day.month,
               current_day.day,current_day.hour)
        f.write(cmd)
    current_day=current_day+datetime.timedelta(hours=6)
    
f.close()


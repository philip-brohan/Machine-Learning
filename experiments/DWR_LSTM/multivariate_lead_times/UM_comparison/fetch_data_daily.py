#!/usr/bin/env python

# Extract the data for a range of days

import os
import subprocess
import datetime


# Function to check if the job is already done for this timepoint
def is_done(year,month,day):
    op_file_name=("%s/Machine-Learning-experiments/datasets/"+
                  "opfc/London/%04d/%02d/%02d/00/000/prmsl.pkl") % (
                            os.getenv('SCRATCH'),
                            year,month,day)
    if os.path.isfile(op_file_name):
        return True
    return False

f=open("fetch2.txt","w+")

start_day=datetime.datetime(2019, 10, 23,  0)
end_day  =datetime.datetime(2018,  9, 25, 23)

current_day=start_day
while current_day>=end_day:
    if is_done(current_day.year,current_day.month,
                   current_day.day):
        current_day=current_day-datetime.timedelta(days=1)
        continue
    cmd=("./get_UM_data_for_day.py --year=%d --month=%d " +
         "--day=%d \n"+
         "./extract_UM_data_for_london.py --year=%d --month=%d " +
         "--day=%d &\n") % (
           current_day.year,current_day.month,
           current_day.day,current_day.year,current_day.month,
           current_day.day)
    f.write(cmd)
    current_day=current_day-datetime.timedelta(days=1)
f.close()


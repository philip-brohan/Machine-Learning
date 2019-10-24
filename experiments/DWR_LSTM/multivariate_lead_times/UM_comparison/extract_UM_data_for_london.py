#!/usr/bin/env python

# Extract and store forecasts and analyses at the location of London
#  from the MO global analysis.

import os
import iris
import pickle
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year",
                    type=int,required=True)
parser.add_argument("--month", help="Integer month",
                    type=int,required=True)
parser.add_argument("--day", help="Day of month",
                    type=int,required=True)
parser.add_argument("--ddir", help="Directory for PP files",
                    default="%s/opfc/tpf" % os.getenv('SCRATCH'))
args = parser.parse_args()
opdir="%s/%04d/%02d" % (args.ddir,args.year,args.month)

# For each analysis run (12-hourly)
for analysis in [0,12]:

    # For each forecast timestep (6-hourly, to 168 hours)
    for fcst in range(0,169,6):

        # Load the PP field
        pp_file="%s/opfc/tpf/%04d/%02d/%02d.pp" % (os.getenv('SCRATCH'),
                                          args.year,args.month,args.day)
        try:
            fp_c=iris.Constraint(forecast_period=fcst)
            frt_c=iris.Constraint(forecast_reference_time=
                   datetime.datetime(args.year,args.month,args.day,analysis))
            f=iris.load(pp_file,fp_c & frt_c)
        except Exception:
            continue

        # For each variable (should have prmsl and t2m)
        for cube in f:

            # interpolate to location
            loc_value=cube.interpolate([('latitude',51.46),
                                        ('longitude',-0.12)],
                        iris.analysis.Linear()).data.item()

            # Pickle the value
            p_dir=("%s/Machine-Learning-experiments/datasets/opfc/"+
                   "London/%04d/%02d/%02d/%02d/%03d") % (os.getenv('SCRATCH'),
                    args.year,args.month,args.day,analysis,fcst)
            if not os.path.isdir(p_dir):
                os.makedirs(p_dir)
            if cube.name()=='air_pressure_at_sea_level':
                pfl_name='prmsl.pkl'
            elif cube.name()=='air_temperature':
                pfl_name='air.2m.pkl'
            else:
                print(cube)
                raise Exception("Unexpected variable")
            pickle.dump(loc_value,open("%s/%s" % (p_dir,pfl_name), 'wb'))

#!/usr/bin/env python

# We're going to work with 20CR2c prmsl fields.
# Retrieve the netCDF files from NERSC and store on $SCRATCH

import datetime
import IRData.twcr as twcr

for year in range(1969,2011):
    # 2c is in 1 year batches so month and day don't matter
    dte=datetime.datetime(year,1,1)
    twcr.fetch('prmsl',dte,version='2c')
    twcr.fetch('air.2m',dte,version='2c')
    twcr.fetch('prate',dte,version='2c')
    #twcr.fetch('z500',dte,version='2c')
    twcr.fetch('uwnd.10m',dte,version='2c')
    twcr.fetch('vwnd.10m',dte,version='2c')

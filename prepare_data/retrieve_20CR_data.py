#!/usr/bin/env python

# We're going to work with 20CR2c prmsl fields.
# Retrieve the netCDF files from NERSC and store on $SCRATCH

import datetime
import IRData.twcr as twcr

for year in [1969,1979,1989,1999,2009]:
    # 2c is in 1 year batches so month and day don't matter
    dte=datetime.datetime(year,1,1)
    twcr.fetch('prmsl',dte,version='2c')

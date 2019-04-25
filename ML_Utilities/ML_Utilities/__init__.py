# (C) British Crown Copyright 2017, Met Office
#
# This code is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#

"""
Module supporting ML experiments - mostly functions for making reanalysis data available to ML models. The idea is to abstract away the data provision to this module, so experiments can concentrate on model building.

The module has two basic functions: The :func:`prepare_data` function takes reanalysis data and reformats it into input files suitable for TensorFlow, and the :func:`dataset` function turns those files into a :obj:`tf.data.Dataset` which can be passed to a model for training or testing.

The first step might be:

.. code-block:: python

    import ML_Utilities

    count=1
    for year in [1969,1979,1989,1999,2009]:

        start_day=datetime.datetime(year,  1,  1,  0)
        end_day  =datetime.datetime(year, 12, 31, 23)

        current_day=start_day
        while current_day<=end_day:
            purpose='training'
            if count%10==0: purpose='test' # keep 1/10 data for testing
            ML_Utilities.prepare_data(current_day,
                                      purpose=purpose,
                                      source='20CR2c',
                                      variable='prmsl')
        # Keep samples > 5 days apart to minimise correlations
        current_day=current_day+datetime.timedelta(hours=126)
        count += 1

That will produce TensorFlow files containing 5 years of 20CRv2c prmsl fields - note that you will need to download the data first - see the `IRData package <https://brohan.org/IRData/>`_.

Then to use those data in an ML model, it's:

.. code-block:: python

    import ML_Utilities

    training_data=ML_Utilities.dataset(purpose='training', 
                                       source='20CR2c', 
                                       variable='prmsl')
    test_data=ML_Utilities.dataset(purpose='test', 
                                       source='20CR2c', 
                                       variable='prmsl')
    # model=(specify here)

    model.fit(x=training_data,
              validation_data=test_data)

You may need to use the ``repeat`` and ``map`` functions on the Datasets to get a long enough dataset for many epochs of training, and to tweak the dataset to the requirements of your model (perhaps to set the array shape). 

"""

from .dataset import *
from .prepare_data import *
from .normalise import *


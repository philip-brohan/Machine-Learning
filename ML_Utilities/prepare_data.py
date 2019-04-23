# (C) British Crown Copyright 2019, Met Office
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

# Prepare a piece of tf data from 20CR - convert to tensor, normalise, save as tf.load-able file.
import os
import iris
import tensorflow as tf
import IRData.twcr as twcr
from .normalise import get_normalise_function

def prepare_data(date,purpose='training',source='20CR2c',variable='prmsl',
                 member=1,normalise=None,opfile=None):
    """Make tf.load-able files, suitably normalised for training ML models 

    Data will be stored in directory $SCRATCH/Machine-Learning-experiments.

    Args:
        date (obj:`datetime.datetime`): datetime to get data for.
        purpose (:obj:`str`): 'training' (default) or 'test'.
        source (:obj:`str`): Where to get the data from - at the moment, only '20CR2c' is supported .
        variable (:obj:`str`): Variable to use (e.g. 'prmsl')
        normalise: (:obj:`func`): Function to normalise the data (to mean=0, sd=1). Function must take an :obj:`iris.cube.cube' as argument and returns a normalised cube as result. If None (default) use a standard normalisation function (see :func:`normalise`.

    Returns:
        Nothing, but creates, as side effect, a tf.load-able file with the normalised data for the given source, variable, and date.

    Raises:
        ValueError: Unsupported source, or can't load the original data, or normalisation failed.

    |
    """
    if opfile is None:
        opfile=("%s/Machine-Learning-experiments/datasets/%s/%s/%s" %
                       (os.getenv('SCRATCH'),source,variable,purpose))
    if not os.path.isdir(os.path.dirname(opfile)):
        os.makedirs(os.path.dirname(opfile))

    ic=twcr.load(variable,datetime.datetime(date.year,date.month,
                                                date.day,date.hour),
                 version=args.version)

    # Reduce to selected ensemble member
    ic=ic.extract(iris.Constraint(member=args.member))
    
    # Normalise (to mean=0, sd=1)
    if normalise is None:
        normalise=get_normalise_function(source,variable)
    ic.data=normalise(ic.data)

    # Convert to Tensor
    ict=tf.convert_to_tensor(ic.data, numpy.float32)

    # Write to tfrecord file
    sict=tf.serialize_tensor(ict)
    tf.write_file(opfile,sict)



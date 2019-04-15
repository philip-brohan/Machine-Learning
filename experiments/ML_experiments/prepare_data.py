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

def prepare_data(date,purpose='training',source='20CR2c',variable='prmsl',normalise=None):
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
        ValueError: Unsupported source, or can't load the original data, or noremalisation failed.

    |
    """
    # File names for the serialised tensors (made by :func:`create_dataset`)
    input_file_dir=("%s/Machine-Learning-experiments/datasets/%s/%s/%s" %
                       (os.getenv('SCRATCH'),source,variable,purpose))
    data_files=glob("%s/*.tfd" % input_file_dir)
    if len(data_files)==0;
        raise ValueError('No prepared data on disc')
    n_steps=len(training_files)
    data_tfd = tf.constant(training_files)

    # Create TensorFlow Dataset objects from the file names
    tr_data = Dataset.from_tensor_slices(data_tfd)
    tr_data = tr_data.shuffle(buffer_size=buffer_size,
                              reshuffle_each_iteration=reshuffle_each_iteration)
    if length is not none:
        nrep=(length//n_steps)+1
        tr_data = tr_data.repeat(nrep)
    
    return tr_data



Making Tensors from archived climate data
=========================================

First we need to get climate data from its archive in native format. For `20CR2c <https://www.esrl.noaa.gov/psd/data/20thC_Rean/>`_ data this is simple using the `IRData package <http://brohan.org/IRData/>`_:

.. literalinclude:: ../../prepare_data/retrieve_20CR_data.py

We can then use `IRData <http://brohan.org/IRData/>`_ again to load a single field as a `Iris cube <https://scitools.org.uk/iris/docs/v2.2/userguide/loading_iris_cubes.html>`_. The cube.data field is a Numpy array that can be directly converted to a tensor, serialised, and written to disc:

.. literalinclude:: ../../prepare_data/make_training_tensor.py

If we do that every 5days+6hours over an extended period, we make a large batch of training data, with fields far apart enough in time to be almost uncorrelated (for MSLP), and evenly distributed through the diurnal and annual cycles. We actually need two such datasets, a large one for training models, and a smaller one for testing them. This script makes the list of commands needed to make all the training and test data, and these commands can be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../prepare_data/makeall.py

As set-up, this system makes tensors from 20CR2c MSLP fields, but it's simple to extend it to other sources and variables.


 

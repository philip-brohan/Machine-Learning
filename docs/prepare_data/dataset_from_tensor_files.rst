Making a tf.data.Dataset from tensor files on disc
==================================================

Once we have the data we're interested in assembled as a set of tensor files, we need to present them as a `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_. The best way I've found to do this is to directly create a Dataset from the file names, use the ``shuffle`` and ``repeat`` methods to make a sequence of files of the required length, and then use the ``map`` method to replace each dataset name with its contents:

.. literalinclude:: ../../prepare_data/setup_dataset.py



 

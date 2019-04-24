Getting climate data into TensorFlow
====================================

To train and use ML models on climate data, we need to set up an input pipeline: To get the data out of its archive, convert it into `tf.tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_ format, and organise it into a `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ which can be passed to the TensorFlow model as a data source.

It's most efficient to do this in two steps: First to extract, convert, and store the data on a fast disc in native TensorFlow format. Then those native files can be repeatedly used in model training.

.. toctree::
   :maxdepth: 1

   archive_to_Tensor
   dataset_from_tensor_files

 

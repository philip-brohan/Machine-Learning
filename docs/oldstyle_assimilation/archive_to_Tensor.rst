Making Tensors of pseudo-observations
=====================================

To train a system that maps observations to analysis fields, we need to generate an appropriate set of observations (as a `tf.tensor <https://www.tensorflow.org/guide/tensors>`_) to pair each analysis field. We'll use pseudo-observations - fake observations made by extracting the values in the analysis field at each of a predefined set of observation locations - as this gets rid of the complications from observation uncertainties and quality control. For this experiment, we are using a fixed observations coverage (from 1916-03-12:06).

We :doc:`already have the analysis fields <../prepare_data/archive_to_Tensor>`, and we can make pseudo-observations from each field by extracting the values at the observation locations, and then export these pseudo-observations as a serialised tensor:

.. literalinclude:: ../../prepare_data/1916_obs/make_training_tensor.py

If we do that every 6hours over an extended period, we make a large batch of training data, and it will be possible to match each target field with an obs tensor. We actually need two such datasets, a large one for training models, and a smaller one for testing them. This script makes the list of commands needed to make all the training and test data, and these commands can be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../prepare_data/1916_obs/makeall.prmsl.py



 

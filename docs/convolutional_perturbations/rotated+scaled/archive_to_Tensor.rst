Making Tensors of rotated and scaled fields
===========================================

We :doc:`already have a script to make the analysis fields as tensors <../../prepare_data/archive_to_Tensor>`, so to make the improved version we can use the same process, except that we rotate each field onto the modified Cassini projection, and re-grid it to a resolution of 79x159 (which is both shrinkable and constructable by stride-2 convolutions).

.. literalinclude:: ../../../prepare_data/rotated_pole/make_training_tensor.py

If we do that every 6hours over an extended period, we make a large batch of training data, and it will be possible to match each target field with an obs tensor. We actually need two such datasets, a large one for training models, and a smaller one for testing them. This script makes the list of commands needed to make all the training and test data, and these commands can be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../../prepare_data/rotated_pole/makeall.prmsl.small.py



 

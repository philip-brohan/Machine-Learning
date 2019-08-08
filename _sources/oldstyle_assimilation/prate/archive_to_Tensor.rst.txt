Making Tensors of pseudo-observations or precipitation rate
===========================================================

This is analagous to :doc:`the same process for prmsl observations <../archive_to_Tensor>` except that it uses a different data source and normalisation.

Script to make a single tensor:

.. literalinclude:: ../../../prepare_data/1916_obs/make_training_tensor.py

Make the commands to generate all the required tensors. To be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../../prepare_data/1916_obs/makeall.prate.small.py



 

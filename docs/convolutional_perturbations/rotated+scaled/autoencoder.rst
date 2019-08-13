An all-convolutional autoencoder with better training data
==========================================================

The :doc:`all-convolutional autoencoder <../all_convolutional/autoencoder>` is a nuisance to use, as the strided convolutions mean it's fussy about resolution, and it's not clear how best to deal with the data boundary conditions. The latter problem is particulary acute when using 20CR2c data, as its longitude splice is at 0/360 and so goes through the UK; so the problem is most acute where I want the results to be best.

Making a convolutional autoencoder which deals correctly with both the periodic and polar boundaries would be tricky. A much easier approach is to re-grid the training data so all the interesting points are in the middle. Regridding to a modified `Cassini projection <https://en.wikipedia.org/wiki/Cassini_projection>`_ with the North pole at 90W on the equator moves the poles to the centre of the crid and puts all the boundary points on the equator. At the same time we can set the field resolution to a value which works nicely with strided convolutions.


.. toctree::
   :titlesonly:
   :maxdepth: 1

   Make rotated and scaled fields for training and validation <archive_to_Tensor>

Then the code is the same as for the :doc:`all-convolutional autoencoder <../all_convolutional/autoencoder>`, except that all the complications of resixzing and padding can be removed:

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/rotated+scaled/autoencoder.py

It works pretty well, with only very minor issues near the boundaries.

.. figure:: ../../../experiments/convolutional_autoencoder_perturbations/rotated+scaled/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom, a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/rotated+scaled/validation/compare_results.py

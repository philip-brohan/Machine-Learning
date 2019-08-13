An all-convolutional autoencoder with more layers
=================================================

The :doc:`all-convolutional autoencoder with optimised training data <../rotated+scaled/autoencoder>` works pretty well. But if we add an additional strided convolutional layer the encoded state would be smaller, so it would be more useful.

Then the code is the same as for the :doc:`all-convolutional autoencoder with optimised training data <../rotated+scaled/autoencoder>`, except that an additional layer is added to both the encoder and decoder:

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/more_layers_4/autoencoder.py

It still works, but the quality of the reconstructed field is somewhat reduced.

.. figure:: ../../../experiments/convolutional_autoencoder_perturbations/more_layers_4/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom, a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/more_layers_4/validation/compare_results.py

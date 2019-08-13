An all-convolutional autoencoder with an explicit latent space
==============================================================

An autoencoder transforms an explicit, high-dimensional representation into lower-dimensional version. It's convenient to represent the compressed state as a vector in an explicit `latent space <https://ai-odyssey.com/2017/02/24/latent-space-visualization%E2%80%8A/>`_.

Then the code is the same as for the :doc:`all-convolutional autoencoder with optimised training data <../rotated+scaled/autoencoder>`, except that an additional, fully-connected, layer is added to form the latent space:

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/latent_space/autoencoder.py

It takes longer to train, and the quality of the reconstruction is somewhat reduced, but with a 100-dimensional latent space the quality is not too bad.

.. figure:: ../../../experiments/convolutional_autoencoder_perturbations/latent_space/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom, a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/latent_space/validation/compare_results.py

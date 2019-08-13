Perturbing the convolutional autoencoder
========================================

So we have :doc:`a convolutional autoencoder that works pretty well <../convolutional_autoencoder/autoencoder>`. Can we make it work even better - either improve the reconstruction accuracy or shrink the encoded state?

.. toctree::
   :maxdepth: 1

   All-convolutional version <all_convolutional/autoencoder>

The all-convolutional architecture is nice, but the periodic boundary conditions are a nuisance to deal with. We can simplify the autoencoder by pre-scaling the data and rotating the interesting features away from the edges of the field.

.. toctree::
   :maxdepth: 1

   With rotated and scaled training data <rotated+scaled/autoencoder>

To make the autoencoder more useful we can try two ways to shrink the encoded state:

.. toctree::
   :maxdepth: 1

   With an extra strided convolution layer <more_layers_4/autoencoder>
   With an explicit latent space <latent_space/autoencoder>

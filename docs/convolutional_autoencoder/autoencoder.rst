A convolutional autoencoder
===========================

Again inspired by `Vikram Tiwari's tf.keras Autoencoder page <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_, we can try a `convolutional <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_ autoencoder. Like the :doc:`deep autoencoder <../deep_autoencoder/autoencoder>`, this has several hidden layers and Leaky-ReLU activations, but the layers are convolutional, rather than fully connected.

Convolutional layers look at spatially-connected clusters of grid-cells together. This introduces a complication for maps of geospatial data, as they have periodic boundary conditions in longitude, and complications at the poles. Also, convolutional layers are usually paired with pooling layers, and an autoencoder using pooling and upsampling requires a grid resolution which divides exactly by the amount of pooling - this example has 3 2x2 pooling layers, so we need both x and y field sizes to be divisible by 8 (2^3). To deal with this I have `written new layers <https://www.tensorflow.org/alpha/guide/keras/custom_layers_and_models>`_ to add and remove periodic-boundary-condition padding along the longitude split, and to resize the data grids. (I'm not worrying about the poles for the moment).

.. literalinclude:: ../../experiments/convolutional_autoencoder/autoencoder.py

For representing MSLP fields, convolution is the way to go: this autoencoder works dramatically better than the :doc:`simple <../simple_autoencoder/autoencoder>` or :doc:`deep <../deep_autoencoder/autoencoder>` versions.

.. figure:: ../../experiments/convolutional_autoencoder/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom, a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../experiments/convolutional_autoencoder/validation/compare_results.py

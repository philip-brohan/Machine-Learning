A deep autoencoder
==================

Again inspired by `Vikram Tiwari's tf.keras Autoencoder page <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_, we can try a deep autoencoder - with four hidden layers. This is fundamentally the same as the :doc:`simple autoencoder <../simple_autoencoder/autoencoder>`, except that it has more hidden layers (also, I've switched to Leaky-ReLU activations as they showed the best combination of performance and speed in :doc:`activation tests <../autoencoder_perturbations/activations/leaky_relu/autoencoder>`).

.. literalinclude:: ../../experiments/deep_autoencoder/autoencoder.py

It's several times slower than the simple autoencoder, and seems to be only slightly improved in quality. 


.. figure:: ../../experiments/deep_autoencoder/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: layer-by-layer from input to output - the layers alternate between fully-connected layers and leaky-relu activation layers
   Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom right, , a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.

Apart from the input and output layers, I don't know how to read the weights colourmaps for the deep autoencoder - plots like this may not be much use.


Script to make the figure
-------------------------

.. literalinclude:: ../../experiments/deep_autoencoder/validation/compare_full.py

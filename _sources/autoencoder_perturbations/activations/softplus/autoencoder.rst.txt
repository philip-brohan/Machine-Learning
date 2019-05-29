Simple autoencoder with softplus activation (rather than tanh)
==============================================================

The original :doc:`simple autoencoder<../../../simple_autoencoder/autoencoder>` used the `tanh activation function <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>`_. What happens if we use `softplus activation <https://www.tensorflow.org/api_docs/python/tf/keras/activations/softplus>`_ instead?

.. literalinclude:: ../../../../experiments/simple_autoencoder_activations/softplus/autoencoder.py

In general this is worse than the :doc:`tanh original <../../../simple_autoencoder/summary>`, and its quality shows a big variation across the map. It works very badly in the low pressure region around Antarctica. I think this is an interaction with the data normalisation - softplus would work much better with data normalised on the range 0-1 rather than to a normal(ish) distribution with mean=0, which is what I'm using here.

.. figure:: ../../../../experiments/simple_autoencoder_activations/softplus/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom right, training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../../experiments/simple_autoencoder_activations/softplus/validation/compare_full.py

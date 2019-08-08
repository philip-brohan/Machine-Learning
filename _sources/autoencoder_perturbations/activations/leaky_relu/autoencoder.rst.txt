Simple autoencoder with leaky relu activation (rather than tanh)
================================================================

The original :doc:`simple autoencoder<../../../simple_autoencoder/autoencoder>` used the `tanh activation function <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>`_. The :doc:`version with relu activation <../relu/autoencoder>` didn't work. What happens if we use `leaky relu activation <https://keras.io/layers/advanced-activations/>`_?

.. literalinclude:: ../../../../experiments/simple_autoencoder_activations/leaky_relu/autoencoder.py

I hoped this would fix the problems with relu, by dealing sensibly with negative inputs, and it does. In fact this seems to have the same advantages as :doc:`elu activation <../elu/autoencoder>`, and runs quite a bit faster.

.. figure:: ../../../../experiments/simple_autoencoder_activations/leaky_relu/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom right, a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../../experiments/simple_autoencoder_activations/leaky_relu/validation/compare_full.py

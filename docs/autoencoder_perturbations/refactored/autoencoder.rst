Simplifying the simple autoencoder
==================================

Most of the :doc:`simple autoencoder script <../../simple_autoencoder/autoencoder>` is code to handle data input. So I created a :doc:`ML_Utilities package <../../package/ML_Utilities>` to handle that (and also to standardise data normalisation). This simplifies the autoencoder script considerably.

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/refactored/autoencoder.py

That change should make almost no difference to the results, and indeed it doesn't (:doc:`see original <../../simple_autoencoder/summary>`):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/refactored/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom right, training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/refactored/validation/compare_full.py

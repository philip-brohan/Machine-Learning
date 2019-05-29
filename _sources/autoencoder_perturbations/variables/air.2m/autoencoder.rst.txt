Simple autoencoder for air.2m fields (instead of prmsl)
=======================================================

We have a :doc:`simple autoencoder for prmsl <../../../simple_autoencoder/autoencoder>`. What happens if we train it on air.2m instead. We need to change the data source and data normalisation, but otherwise the model is exactly the same:

.. literalinclude:: ../../../../experiments/simple_autoencoder_variables/air.2m/autoencoder.py

It captures much more of the variance than the prmsl version (compare :doc:`original <../../../simple_autoencoder/summary>`). Even though it's only using half it's neurons (why)?:

.. figure:: ../../../../experiments/simple_autoencoder_variables/air.2m/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample 2m temperature field: Original in red, after passing through the autoencoder in blue.
   Bottom right, training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../../experiments/simple_autoencoder_variables/air.2m/validation/compare_full.py

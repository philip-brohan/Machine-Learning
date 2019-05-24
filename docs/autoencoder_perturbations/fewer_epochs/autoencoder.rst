Simple autoencoder with fewer epochs of training
================================================

The original :doc:`simple autoencoder<../../simple_autoencoder/autoencoder>` was trained for 100 epochs on 2566 MSLP fields. What happens if we reduce the number of training epochs (to 10).

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/fewer_epochs/autoencoder.py

Less training should produce a poorer quality reconstruction (in less time), and it does (:doc:`see original <../../simple_autoencoder/summary>`):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/fewer_epochs/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom right, training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/fewer_epochs/validation/compare_full.py

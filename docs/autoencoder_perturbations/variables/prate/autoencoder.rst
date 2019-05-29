Simple autoencoder for precipitation fields (instead of prmsl)
==============================================================

We can do :doc:`prmsl <../../../simple_autoencoder/autoencoder>`, and :doc:`air.2m <../air.2m/autoencoder>`. Let's take on a challenge, and try precipitation.

Normalisation is difficult - I chose (rather arbitrarily), to take the square root of the rate and then multiply by 100. This gives a value between 0 and about 3.5 - with a big spike at 0.

I switched from tanh to elu activation - tanh seemed less appropriate for variables all one side of 0, but again - arbitrary choice.

I kept the RMS error metric. This is probably a poor choice - the model residuals will not be even approximately normally distributed, but it's not obvious to me what would be better.

Otherwise - proceed as before:

.. literalinclude:: ../../../../experiments/simple_autoencoder_variables/prate/autoencoder.py

Somewhat to my suprise, this does work to an extent (compare :doc:`prmsl <../../../simple_autoencoder/summary>`, :doc:`air.2m <../air.2m/autoencoder>`). It's only using 12 neurons(why)?, it's overfitted (best at 10 epochs, why?), but it's not hopeless. More cunning approaches to normalisation, and model parameters, might make a big improvement.

.. figure:: ../../../../experiments/simple_autoencoder_variables/prate/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample precipitation field (note, after square-root transform): Original in red, after passing through the autoencoder in blue.
   Bottom right, training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../../experiments/simple_autoencoder_variables/prate/validation/compare_full.py

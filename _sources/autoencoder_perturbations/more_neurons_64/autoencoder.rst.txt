Simple autoencoder with more neurons (64 rather than 32)
========================================================

The original :doc:`simple autoencoder<../../simple_autoencoder/autoencoder>` had 32 neurons in its fully-connected hidden layer. What happens if we increase this number to 64?

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/more_neurons_64/autoencoder.py

It should train more slowly, and it does; otherwise, I was not sure what to expect except that I hoped for a notable improvement. Actually, it seems little different from the 32-neuron version. The validation loss at 100 epochs is smaller, but only just (0.19 at 2 s.f. for both 32 and 64):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/more_neurons_64/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:06. Original in top panel - after passing through the original simple autoencoder in middle panel, and after passing through the autoencoder with more neurons (64 rather than 32) in bottom panel. Note that the contour spacings are different between the panels.

|


.. toctree::
   :maxdepth: 1

   validation

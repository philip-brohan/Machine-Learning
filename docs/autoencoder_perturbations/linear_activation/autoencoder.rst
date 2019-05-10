Simple autoencoder with linear activation (rather than tanh)
============================================================

The original :doc:`simple autoencoder<../../simple_autoencoder/autoencoder>` used the `tanh activation function <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>`_. What happens if we use `linear activation <https://www.tensorflow.org/api_docs/python/tf/keras/activations/linear>`_ instead?

.. literalinclude:: ../../../experiments/simple_autoencoder_activations/linear/autoencoder.py

It should train faster, and it does; otherwise, I was not sure what to expect except that it should be notably worse in some respect. Actually, it seems little different from the 32-neuron version. The validation loss at 100 epochs is bigger (0.21 for 8 neurons instead of 0.19 for 32), but the effect is small.:

.. figure:: ../../../experiments/simple_autoencoder_activations/linear/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:06. Original in top panel - after passing through the original simple autoencoder in middle panel, and after passing through the autoencoder with linear activation (rather than tanh) in bottom panel. Note that the contour spacings are different between the panels.

|


.. toctree::
   :maxdepth: 1

   validation

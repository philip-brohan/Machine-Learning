Getting started - a minimal autoencoder
=======================================

I like the idea of `autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`_ - part of the difficulty of working with GCM output is that there is so *much* of it, an automatic system for reducing it to a more manageable size is very appealing. So I looked for a very simple TensorFlow autoencoder example, and chose the first one from `Vikram Tiwari's tf.keras Autoencoder page <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_.

To follow that example, but using 20CR2c MSLP fields instead of MNIST as the target, I made three changes:
   - I used the :doc:`20CR2c data source <../prepare_data/dataset_from_tensor_files>`.
   - I changed the activation from ``relu`` to ``tanh`` as the normalised mslp data is spread around 0 rather than on the range 0-1.
   - I changed the loss metric to RMS from ``binary_crossentropy`` (because this is a regression problem, not a classification).

.. literalinclude:: ../../experiments/simple_autoencoder/autoencoder.py

This runs nicely on my Linux desktop - takes about s/epoch (with 315 training fields) and uses 6 cores (of 8). The validation loss falls from around 0.5 to 0.19 over 100 epochs. So it's learning something.

To see exactly what it does, we can compare an original MSLP field with the same field passed through the autoencoder (after 100 epochs training):

.. figure:: ../../experiments/simple_autoencoder/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:12. Original in top panel - after passing through the autoencoder in bottom panel. 

|

 The result is not bad, considering the simplicity of the model (32 neurons, only a few minutes training time - it gets the subtropical highs and the storm tracks.

Because the model is so simple, we can also examine the weights directly:

.. figure:: ../../experiments/simple_autoencoder/introspection/weights.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Weights after 100 epochs training. The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the botton are the output layer weights, arranged in the same way.

The weights show the model is behaving sensibly, I expected it to converge on a set of output patterns similar to `EOFs <https://en.wikipedia.org/wiki/Empirical_orthogonal_functions>`_ of the pressure field, a matching set of input patterns, and a range of hidden-layer weights indicating which patterns were typicaly bigger. We do get something that looks like that. 

So it does work, though not well enough to be useful. Success is then *merely* a matter of specifying and training a better model.

.. toctree::
   :maxdepth: 1

   validation
   weights
   summary

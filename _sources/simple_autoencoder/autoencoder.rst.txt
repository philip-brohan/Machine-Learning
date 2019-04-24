Getting started - a minimal autoencoder
=======================================

I like the idea of `autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`_ - part of the difficulty of working with GCM output is that there is so *much* of it, an automatic system for reducing it to a more manageable size is very appealing. So I looked for a very simple TensorFlow autoencoder example, and chose the first one from `Vikram Tiwari's tf.keras Autoencoder page <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_.

To follow that example, but using 20CR2c MSLP fields instead of MNIST as the target, I made three changes:
   - I used the :doc:`20CR2c data source <../prepare_data/dataset_from_tensor_files>`.
   - I changed the activation from ``relu`` to ``sigmoid`` as the normalised mslp data is spread around 0 rather than on the range 0-1.
   - I changed the loss metric to RMS from ``binary_crossentropy`` (because I don't know what binary crossentropy is).

.. literalinclude:: ../../experiments/simple_autoencoder/autoencoder.py

This runs nicely on my Linux desktop - takes about 2s/epoch (with 315 training fields) and uses 6 cores (of 8). The validation loss falls from around 0.5 to 0.19 over 100 epochs. So it's learning something.

To see exactly what it does, we can compare an original MSLP field with the same field passed through the autoencoder (after 100 epochs training):

.. figure:: ../../experiments/simple_autoencoder/validation/comparison_2009-03-12:12.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:12. Original in top panel - after passing through the autoencoder in bottom panel. Note that the contour spacings are different between the panels.

|

We might expect an underpowered autoencoder minimising RMS error to ignore the input field and just guess a climatology every time. That is what happens here - the output is almost the same whatever the input field. The climatology is not bad though, considering the simplicity of the model (32 neurons, less than 2 minutes training time - it gets the subtropical highs and the storm tracks.

So it does work, though not well enough to be useful. Success is then *merely* a matter of specifying and training a better model.

.. toctree::
   :maxdepth: 1

   validation

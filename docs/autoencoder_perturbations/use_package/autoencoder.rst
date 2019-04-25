Simplifying the simple autoencoder
==================================

Most of the :doc:`simple autoencoder script <../../simple_autoencoder/autoencoder>` is code to handle data input. So I created a :doc:`ML_Utilities package <../../package/ML_Utilities>` to handle that (and also to standardise data normalisation). This simplifies the autoencoder script considerably.

.. literalinclude:: ../../../experiments/simple_autoencoder_with_utilities_library/autoencoder.py

That change should make almost no difference to the results, and indeed it doesn't:

.. figure:: ../../../experiments/simple_autoencoder_with_utilities_library/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:06. Original in top panel - after passing through the original simple autoencoder in middle panel, and after passing through the refactored autoencoder in bottom panel. Note that the contour spacings are different between the panels.

|


.. toctree::
   :maxdepth: 1

   validation

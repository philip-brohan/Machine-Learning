Simple autoencoder with less training data
==========================================

The original :doc:`simple autoencoder<../../simple_autoencoder/autoencoder>` was trained for 100 epochs on 315 MSLP fields. What happens if we reduce the number of training fields (to 30).

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/less_training_data/autoencoder.py

Less training data should produce a poorer quality reconstruction, and it does (it should also run through the training much faster, and it does that too):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/less_training_data/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:06. Original in top panel - after passing through the original simple autoencoder in middle panel, and after passing through the autoencoder with reduced training data in bottom panel. Note that the contour spacings are different between the panels.

|


.. toctree::
   :maxdepth: 1

   validation
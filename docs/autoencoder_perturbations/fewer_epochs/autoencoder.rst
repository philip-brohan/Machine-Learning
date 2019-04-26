Simple autoencoder with fewer epochs of training
================================================

The original :doc:`simple autoencoder<../../simple_autoencoder/autoencoder>` was trained for 100 epochs on 315 MSLP fields. What happens if we reduce the number of training epochs (to 10).

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/fewer_epochs/autoencoder.py

Less training should produce a poorer quality reconstruction, and it does (it should also complete the training in less time, and it does that too):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/fewer_epochs/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:06. Original in top panel - after passing through the original simple autoencoder in middle panel, and after passing through the autoencoder with fewer training epochs in bottom panel. Note that the contour spacings are different between the panels.

|


.. toctree::
   :maxdepth: 1

   validation

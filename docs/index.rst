Philip's Machine Learning Experiments
=====================================

Climate research is a data-rich field. We have terrabytes of in-situ observations, and many petabytes each of satellite observations, reanalyses, forecasts, and simulations. Modern `Machine Learning (ML) <https://en.wikipedia.org/wiki/Machine_learning>`_ methods promise powerful new ways to analyse and improve both forecasts and reconstructions; if we can learn to use them effectively.

This page documents my attempts to learn and use ML methods. I chose to use `TensorFlow <https://www.tensorflow.org/>`_ (as it's powerful, popular, and open), and to work with data from the `20th Century Reanalysis <https://www.esrl.noaa.gov/psd/data/20thC_Rean/>`_ (as it's open, accessible, and I already have `software for working with it <https://brohan.org/IRData/>`_).

.. toctree::
   :maxdepth: 1

   Getting climate data into TensorFlow <prepare_data/prepare_data>

.. toctree::
   :maxdepth: 1

   Getting started - a simple autoencoder <simple_autoencoder/autoencoder>
   Perturbing the simple autoencoder <autoencoder_perturbations/perturbations>
   A deep autoencoder <deep_autoencoder/autoencoder>
   A convolutional autoencoder <convolutional_autoencoder/autoencoder>
   Perturbing the convolutional autoencoder <convolutional_perturbations/perturbations>
 
.. toctree::
   :maxdepth: 1

   Old-style assimilation <oldstyle_assimilation/assimilation>
   Old-style assimilation of precipitation <oldstyle_assimilation/prate/assimilation>
  
.. toctree::
   :titlesonly:
   :maxdepth: 1

   ML Utilities package <package/ML_Utilities>
   Small Print <credits>

This document and the data associated with it are crown copyright (2019) and licensed under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. All code included is licensed under the terms of the `GNU Lesser General Public License <https://www.gnu.org/licenses/lgpl.html>`_.

Oldstyle Assimilation
=====================

The :doc:`convolutional autoencoder <../convolutional_autoencoder/autoencoder>`, works nicely, but it does nothing other than reproduce its input. Can we modify it to make the same output (a 20CRv2c-like MSLP field) from some other input? That would make it a `Generative Model <https://en.wikipedia.org/wiki/Generative_model>`_, and it could become practically useful.

In particular I would like to make a 20CRv2c-like MSLP field from observations of MSLP. This is essentially what `20CR <https://www.esrl.noaa.gov/psd/data/20thC_Rean/>`_ does, so a machine learning system trained to do this would be a cheap approximator to the expensive 20CR system. The general idea is to keep the encoder half of the :doc:`convolutional autoencoder <../convolutional_autoencoder/autoencoder>`, but to replace the encoder half - so we are constructing the encoded MSLP state from a set of observations, rather than from the decoded target state, and then decoding that state back into the full MSLP field as before. This is not quite data assimilation as we are not assimilating observations into a forecast prior - but it attacks the same problem. I'm calling it 'oldstyle assimilation' as I think it's essentially what `Fitzroy <https://en.wikipedia.org/wiki/Robert_FitzRoy>`_ & co. did with the original `Daily Weather Reports <https://digital.nmla.metoffice.gov.uk/collection_86058de1-8d55-4bc5-8305-5698d0bd7e13/>`_: taking a set of observations directly, and drawing a pressure map based on those observations.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   First we need to make pseudo-observations files for training and validation <archive_to_Tensor>
   
Then we can train a model to make an MSLP field from the pseudo-observations.

.. literalinclude:: ../../experiments/oldstyle_assimilation/1916_obs_prmsl/assimilate.py

Considering the simplicity and speed of the system, this works none too badly: The target field is fairly well reproduced in regions close to observations, and in regions far from observations, the system reverts to climatology.

.. figure:: ../../experiments/oldstyle_assimilation/1916_obs_prmsl/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Target in red, as reconstructed from observations (black dots) in blue.
   Bottom, a scatterplot of target v. reconstructed pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../experiments/oldstyle_assimilation/1916_obs_prmsl/validation/compare_results.py
		 
		    

Oldstyle Assimilation with precipitation observations
=====================================================

The :doc:`oldstyle assimilation <../assimilation>` makes no assumptions about what the observations to be assimilated are: It will take anything as long as it is possible to infer the MSLP output field from them. So we should be able to use exactly the same method (and code) to assimilate (pseudo-)observations of precipitation (rather than MSLP).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   First we need to make precipitation pseudo-observations files for training and validation <archive_to_Tensor>
   
Then we can train a model to make an MSLP field from the pseudo-observations. Exactly the same code as for the MSLP observations version (except for the data source and the normalisation).

.. literalinclude:: ../../../experiments/oldstyle_assimilation/1916_obs_prate/assimilate.py

This actually works. The reconstruction is nothing like as good as with :doc:`pressure observations <../assimilation>`, but that's as expected - there's much less information about the pressure field in precipitation observations than pressure observations.

.. figure:: ../../../experiments/oldstyle_assimilation/1916_obs_prate/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Target in red, as reconstructed from precipitation observations (black dots) in blue.
   Bottom, a scatterplot of target v. reconstructed pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/oldstyle_assimilation/1916_obs_prate/validation/compare_results.py
		 
		    

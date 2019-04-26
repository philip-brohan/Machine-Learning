Perturbing the simple autoencoder
=================================

So we have :doc:`a toy autoencoder that works <../simple_autoencoder/autoencoder>`, if not very usefully. Toys are for playing with; what happens if we tweak it a bit - change the parameters?

The first change is a refactoring - move the data provision and normalisation code into a module. That will clean up the code somewhat, and should make no substantial difference to the results.

.. toctree::
   :maxdepth: 1

   Refactoring <refactored/autoencoder>

Then we can start experimenting in earnest:

.. toctree::
   :maxdepth: 1

   Less training on the data <fewer_epochs/autoencoder>
   Less data to train on <less_training_data/autoencoder>

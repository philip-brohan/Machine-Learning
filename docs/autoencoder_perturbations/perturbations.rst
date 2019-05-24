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

.. toctree::
   :maxdepth: 1

   Fewer neurons in the hidden layer (8 rather than 32) <fewer_neurons_8/autoencoder>
   Fewer neurons in the hidden layer (2 rather than 32) <fewer_neurons_2/autoencoder>
   More neurons in the hidden layer (64 rather than 32) <more_neurons_64/autoencoder>

.. toctree::
   :maxdepth: 1

   With L1 regularization <regularized/autoencoder>

.. toctree::
   :maxdepth: 1

   With linear activations (rather than tanh) <activations/linear/autoencoder>

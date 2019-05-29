Simple autoencoder introspection script
=======================================

Each of the 32 neurons in the hidden layer has a weight of its own, an array of 91*180 weights linking it to the inputs (one for each grid cell in the 20CR2c data), and an array of 91*180 weights linking it to the outputs. We can plot all these in a single figure.

.. literalinclude:: ../../experiments/simple_autoencoder/introspection/view_weights.py

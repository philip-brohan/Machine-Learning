Regularizing the simple autoencoder
===================================

The :doc:`simple autoencoder <../../simple_autoencoder/autoencoder>` has 32 neurons in its hidden layer. If this were a linear regression problem with 32 features, I'd worry about overfitting because of colinearity between the features in the training set, and apparently similar things can happen in neural network models. The traditional defence against such overfitting is `regularization <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_, and (again following `Vikram Tiwari <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_) we can add a `tf.keras.regularizers <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers>`_ entry to our hidden layer to reduce possible overfitting here.

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/regularized/autoencoder.py

We already know, however, that this model is not overfitting badly, because we have both a training and a test dataset, and the loss metrics are about the same. If the model were overfitting, the validation loss would be much bigger than the training loss. So here regularization has little effect (:doc:`see original <../../simple_autoencoder/summary>`):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/regularized/validation/comparison_full.png
   :width: 95%
   :align: center
   :figwidth: 95%

   On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.
   Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom right, training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/regularized/validation/compare_full.py

Regularizing the simple autoencoder
===================================

The :doc:`simple autoencoder <../../simple_autoencoder/autoencoder>` has 32 neurons in its hiddel layer. If this were a linear regression problem with 32 features, I'd worry about overfitting because of colinearity between the features in the training set, and apparently similar things can happen in neural network models. The traditional defence against such overfitting is `regularization <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_, and (again following `Vikram Tiwari <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_) we can add a `tf.keras.regularizers <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers>`_ entry to our hidden layer to reduce possible overfitting here.

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/regularized/autoencoder.py

We already know, however, that this model is not overfitting badly, because we have both a training and a test dataset, and the loss metrics are about the same. If the model were overfitting the validation loss would be much bigger than the training loss. Regularisation should slightly degrade the model fit to the training data, and it does (training loss metric at 100 epochs goes from 0.1953 to 0.1985), and it might help the fit to the validation data if the validation loss were notably worse than the training loss, but here it isn't, so regularisation has little effect (validation loss metric at 100 epochs goes from 0.1910 to 0.1942):

.. figure:: ../../../experiments/simple_autoencoder_perturbations/regularized/validation/comparison_2009-03-12:06.png
   :width: 95%
   :align: center
   :figwidth: 95%

   MSLP contours from 20CR2c for 2009-03-12:06. Original in top panel - after passing through the original simple autoencoder in middle panel, and after passing through the regularized autoencoder in bottom panel. Note that the contour spacings are different between the panels.

|


.. toctree::
   :maxdepth: 1

   validation

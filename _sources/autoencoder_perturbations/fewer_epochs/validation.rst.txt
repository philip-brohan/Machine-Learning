Perturbed autoencoder validation script
========================================

To test the autoencoder with fewer epochs of training we need to load the trained model (with `tf.keras.models.load_model <https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model>`_), load an original MSLP field and convert it to a tensor (exactly as we did when :doc:`building the model training data <../../prepare_data/archive_to_Tensor>`), run the test field through each of the original and perturbed autoencoders (with the model ``predict_on_batch`` function), convert the autoencoded tensors back into the original units (reverse the normalisation), and then plot the original and encoded fields (with the `Meteographica package <http://brohan.org/Meteorographica/>`_).

.. literalinclude:: ../../../experiments/simple_autoencoder_perturbations/fewer_epochs/validation/compare.py

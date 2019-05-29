Video diagnostics of model training
===================================

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/338277430?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    <tr><td><center>Autencoder model state (left) validation (right)</center></td></tr>
    </table>
    </center>

On the left, the model weights: The boxplot in the centre shows the weights associated with each neuron in the hidden layer, arranged in order, largest to smallest. Negative weights have been converted to positive (and the sign of the associated output layer weights switched accordingly). The colourmaps on top are the weights, for each hidden layer neuron, for each input field location (so a lat:lon map). They are aranged in the same order as the hidden layer weights (so if hidden-layer neuron 3 has the largest weight, the input layer weights for neuron 3 are shown at top left). The colourmaps on the bottom are the output layer weights, arranged in the same way.

Top right, a sample pressure field: Original in red, after passing through the autoencoder in blue.

Bottom right, training progress: Loss v. no. of training epochs (but note that one epoch here is different (fewer training fields) than one epoch in the other examples).

|

That means saving the model state after each epoch, and having more, shorter epochs - we can modify the autoencoder script to do this.

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/autoencoder.py

Then we need a script to make a :doc:`summary plot <../../simple_autoencoder/summary>` at each epoch:

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/validation_video/compare_full.py

To make the video, it is necessary to run the script above hundreds of times - giving an image after each epoch of training. This script makes the list of commands needed to make all the images, which can be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/validation_video/runall_full.py

To turn the thousands of images into a movie, use `ffmpeg <http://www.ffmpeg.org>`_

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/validation_video/make_video_full.sh


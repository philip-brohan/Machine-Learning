Generating video diagnostics
============================

The :doc:`simple autoencoder summary plot <../../simple_autoencoder/summary>` gives a nice representation of the model state and output quality after training. I'd like to show the same representation, as it varies through training, as a video. 

|

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

.. toctree::
   :maxdepth: 1

   summary_video

|

I'd also like a video of the presure field evolution - both original and after passing through the (fully-trained) autoencoder:

|

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/338276308?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    <tr><td><center>Mean sea-level pressure contours from 20CRv2c (red), and after passing through the autoencoder (blue).</center></td></tr>
    </table>
    </center>


.. toctree::
   :maxdepth: 1

   comparison_video


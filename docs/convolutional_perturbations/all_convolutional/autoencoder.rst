An all-convolutional autoencoder
================================

The original :doc:`convolutional autoencoder <../../convolutional_autoencoder/autoencoder>` alternates convolutional layers with max-pooling layers or upsampling layers to reduce or expand the state. 

`Springenberg et al. <https://arxiv.org/abs/1412.6806>`_ pointed out that you could do the same using only convolutional layers - using strided convolutiona instead of convolutions and max-pooling, and strided transpose convolutions instead of convolutions and up-sampling. This ought to work better, because the layers can learn up- and down-sampling methods instead of imposing them, and `Radford et al. <https://arxiv.org/abs/1511.06434>`_ found that it was a big improvement.

A complication is the requirement to preserve periodic boundary conditions in latitude, and to handle variable grid sizes:
* A 3x3 convolution reduces an mxn grid to (m-2)x(n-2), and it's desirable to have an odd number of grid points to keep things symmetric. So before doing a 3x3 convolution we should have m and n both odd and add an extra layer of cells around the edge - periodic boundary conditions in longitude, and reflection padding in latitude. The GlobePadLayer applies this padding.
* So start with an m*n layer (m,n both odd). Pad -> (m+2)*(n+2). 3x3 convolution -> back to m*n.
* A 3x3 convolution with stride 2 reduces an mxn grid to ((m-1)/2)x((n-1)/2).
* So, start with an m*n layer (m,n both odd). 3x3 convolution with stride 2 -> ((m-1)/2)x((n-1)/2), both odd. 
* A 3x3 transpose convolution with stride 2 upscales a m*n grid to ((m-2)*2)x((n-2)*2) always an even number.
So in the encoding phase, m and n are both always odd, if we start with a 129x257 grid (generally 2^n+1x2^(n+1)+1 for approximately square pixels). 3 reduction layers (3x3 convolution with stride 2) will take this down to 17x33.
Decoding that 3 times (3x3 transpose convolution with stride 2) will bump it back up to 108x212. So we need to resize at both ends.

The change is simple to make:

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/all_convolutional/autoencoder.py

It does indeed work markedly better than the the :doc:`original convolutional version <../../convolutional_autoencoder/autoencoder>`.

.. figure:: ../../../experiments/convolutional_autoencoder_perturbations/all_convolutional/validation/comparison_results.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Top, a sample pressure field: Original in red, after passing through the autoencoder in blue.
   Bottom, a scatterplot of original v. encoded pressures for the sample field, and a graph of training progress: Loss v. no. of training epochs.


Script to make the figure
-------------------------

.. literalinclude:: ../../../experiments/convolutional_autoencoder_perturbations/all_convolutional/validation/compare_results.py

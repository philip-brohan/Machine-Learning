Video diagnostics of model output
=================================

What does the time-evolution of the autoencoded pressure field look like? 

|

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/338276308?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    <tr><td><center>Mean sea-level pressure contours from 20CRv2c (red), and after passing through the autoencoder (blue).</center></td></tr>
    </table>
    </center>


|


This takes a script to make a contour plot of original and autoencoded pressures at a specified hour:

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/validation_video/compare.py

To make the video, it is necessary to run the script above hundreds of times - every 15 minutes for a month. This script makes the list of commands needed to make all the images, which can be run `in parallel <http://brohan.org/offline_assimilation/tools/parallel.html>`_.

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/validation_video/runall_compare.py

To turn the thousands of images into a movie, use `ffmpeg <http://www.ffmpeg.org>`_

.. literalinclude:: ../../../experiments/simple_autoencoder_instrumented/validation_video/make_video_compare.sh


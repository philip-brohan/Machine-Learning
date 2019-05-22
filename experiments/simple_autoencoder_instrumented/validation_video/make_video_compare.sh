#!/bin/bash

ffmpeg -r 24 -pattern_type glob -i /scratch/hadpb/Machine-Learning-experiments/simple_autoencoder_instrumented/images/2\*.png -c:v libx264 -preset slow -tune animation -profile:v high -level 4.2 -pix_fmt yuv420p -crf 25 -c:a copy compare.mp4


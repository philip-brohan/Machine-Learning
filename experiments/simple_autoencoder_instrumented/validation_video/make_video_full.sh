#!/bin/bash

ffmpeg -r 12 -pattern_type glob -i /scratch/hadpb/Machine-Learning-experiments/simple_autoencoder_instrumented/images/comparison_\*.png -c:v libx264 -preset slow -tune animation -profile:v high -level 4.2 -pix_fmt yuv420p -crf 25 -c:a copy full.mp4


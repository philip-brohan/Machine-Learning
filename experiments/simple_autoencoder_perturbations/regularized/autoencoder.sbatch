#!/bin/ksh -l

#SBATCH --output=autoencoder.out
#SBATCH --qos=normal
#SBATCH --ntasks=6
#SBATCH --ntasks-per-core=2
#SBATCH --mem=5000
#SBATCH --time=40:00

./autoencoder.py


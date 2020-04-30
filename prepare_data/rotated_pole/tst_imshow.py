
import tensorflow as tf

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch

file_name="/Users/philip/LocalData/Machine-Learning-experiments/datasets/rotated_pole/20CR2c/prmsl/training/1969-01-01:00.tfd"
sict=tf.io.read_file(file_name) # serialised
ict=tf.io.parse_tensor(sict,np.float32)
fig, ax = plt.subplots()
Z=ict.numpy()
im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
               origin='lower', extent=[-2, 2, -1, 1],
               vmax=3, vmin=-3)

plt.show()

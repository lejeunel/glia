from os.path import join as pjoin
from skimage import io, segmentation
import matplotlib.pyplot as plt
from pyglia import libglia
import numpy as np
import os
import sys

if __name__ == "__main__":

    im_path = sys.argv[-1]
    path = os.path.split(im_path)[0]
    im_fname = os.path.splitext(os.path.split(im_path)[1])[0]

    img = io.imread(im_path)

    watershed = libglia.watershed(img, level=10., relabel=True)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(watershed)
    fig.show()

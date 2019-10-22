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

    path_rgb = pjoin(path, '{}_rgb.png'.format(im_fname))
    path_real = pjoin(path, '{}_real.png'.format(im_fname))
    path_np_rgb = pjoin(path, '{}_np_rgb.png'.format(im_fname))
    path_np_real = pjoin(path, '{}_np_real.png'.format(im_fname))

    # Test conversions
    libglia.test_conversion(img,
                            im_path,
                            path_rgb,
                            path_real,
                            path_np_rgb,
                            path_np_real)

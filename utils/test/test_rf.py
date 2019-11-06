from os.path import join as pjoin
from skimage import io, segmentation
import matplotlib.pyplot as plt
from glia import libglia
import numpy as np
import os
import sys

if __name__ == "__main__":

    n, D = 20, 4
    X = np.random.randn(n, D)
    Y = np.random.choice([1, 0], n)

    print('testing feature conversion')
    print(X)
    libglia.test_conversion_shogun_feats(X)

    print('testing label conversion')
    print(Y)
    libglia.test_conversion_shogun_labels(Y)

    print('testing rf')
    libglia.train_rf(X, Y, 2, 0, 0.7, 0, False)

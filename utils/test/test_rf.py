from os.path import join as pjoin
from skimage import io, segmentation
import matplotlib.pyplot as plt
from glia import libglia
import numpy as np
import os
import sys

if __name__ == "__main__":

    n, D = 10, 4
    X = np.random.randn(n, D)
    Xarea0 = np.abs(np.random.uniform(0.01, 0.05, size=n))
    Xarea1 = np.abs(np.random.normal(0.01, 0.05, size=n))
    X = np.concatenate((Xarea0[..., None], X), axis=1)
    X = np.concatenate((Xarea1[..., None], X), axis=1)
    Y = np.random.choice([1, 0], n)

    median = np.median(np.concatenate((Xarea0, Xarea1)))
    print('median numpy: {}'.format(median))

    print(X)

    # print('testing feature conversion')
    # print(X)
    # libglia.test_conversion_shogun_feats(X)

    # print('testing label conversion')
    # print(Y)
    # libglia.test_conversion_shogun_labels(Y)

    print('testing rf')
    hmt = libglia.hmt(3, 100, 0, 0.7, True)
    hmt.train_rf(X, Y)


#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""
Cython function for calculating the pairwise pixel confusion matrix.

Jeffrey Bush, 2018, NCMIR, UCSD
"""

from libc.stdint cimport uint64_t

def count_pairs(uint64_t[::1] im, uint64_t[::1] gt):
    """
    Count how many of each pair of values are found in the two arrays. If there are n values in im
    and m values in gt then the return value is a 2D array that is n x m.
    """
    from numpy import zeros
    cdef Py_ssize_t i
    n, m = int(im.base.max())+1, int(gt.base.max())+1
    cdef uint64_t[:, ::1] pairs = zeros((n, m), dtype='uint64')
    with nogil:
        for i in range(im.shape[0]): pairs[im[i], gt[i]] += 1
    return pairs.base

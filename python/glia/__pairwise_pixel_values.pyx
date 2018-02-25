#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: language=c++

"""
Cython function for calculating the pairwise pixel confusion matrix.

Jeffrey Bush, 2018, NCMIR, UCSD
"""

ctypedef Py_ssize_t ssize_t

def pairwise_pixel_values(unsigned long[::1] im, unsigned long[::1] gt):
    cdef ssize_t N = im.shape[0], i, j, TP = 0, FP = 0, FN = 0, TN = 0
    cdef bint im_eq, gt_eq
    if N != gt.shape[0]:
        raise ValueError('Image and Ground Truth not the same shape')
    with nogil:
        for i in range(N):
            for j in range(i+1, N):
                im_eq = im[i] == im[j]
                gt_eq = gt[i] == gt[j]
                if im_eq and gt_eq: TP += 1
                elif im_eq and not gt_eq: FP += 1
                elif not im_eq and gt_eq: FN += 1
                else: TN += 1
    return TP, FP, FN, TN

#!/usr/bin/env python2
"""
Statistics utilities. Measures the Accuracy, F-value, and G-mean using pairwise pixel values.

Jeffrey Bush, 2018, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

def main():
    from argparse import ArgumentParser
    from itertools import izip
    from ._utils import open_image_stack_lbl
    from pysegtools.images.filters.label import RelabelImageStack
    from math import sqrt
    from numpy import zeros

    # Setup parser
    parser = ArgumentParser(description="Measures the Accuracy, F-value, and G-mean using pairwise pixel values.",
                            fromfile_prefix_chars='@')
    parser.add_argument('image-stack', type=open_image_stack_lbl,
                        help='the image stack of calculated labels, this accepts any grayscale '+
                        'image stacks that can be given to `imstack -L` except that it may need to be in quotes')
    parser.add_argument('ground-truth', type=open_image_stack_lbl,
                        help='the image stack of ground truth labels, this accepts any grayscale '+
                        'image stacks that can be given to `imstack -L` except that it may need to be in quotes')
    parser.add_argument('-3', dest='threeD', action='store_true',
                        help='if given data is assumed to be 3D and labels are counted across slices')
    parser.set_defaults(threeD=False)


    # Get arguments
    args = parser.parse_args()
    _,ims = getattr(args, 'image-stack')
    _,gts = getattr(args, 'ground-truth')
    threeD = args.threeD

    from datetime import datetime

    # Make sure we have grayscale images
    ims = RelabelImageStack(ims, per_slice=not threeD)
    gts = RelabelImageStack(gts, per_slice=not threeD)

    # Count up confusion matrix
    if threeD:
        im = ims.stack
        gt = gts.stack
        total_im_labels = im.max()
        total_gt_labels = gt.max()
        confusion_matrix = pairwise_pixel_confusion_matrix(im, gt)
    else:
        total_im_labels = 0
        total_gt_labels = 0
        confusion_matrix = zeros((2,2), dtype='uint64')
        for im,gt in izip((im.data for im in ims), (gt.data for gt in gts)):
            total_im_labels += im.max()
            total_gt_labels += gt.max()
            confusion_matrix += pairwise_pixel_confusion_matrix(im, gt)

    # Calculate metrics
    ((TP,FP),(FN,TN)) = confusion_matrix
    precision,recall,specificity = TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    Fvalue = 2*precision*recall / (precision + recall)
    Gmean = sqrt(recall*specificity)

    # Output
    print("Total Calculated Labels:   %d"%total_im_labels)
    print("Total Ground Truth Labels: %d"%total_gt_labels)
    print("TP: %d  FP: %d  FN: %d  TN: %d"%(TP,FP,FN,TN))
    print("Accuracy: %f"%accuracy)
    print("F-value:  %f"%Fvalue)
    print("G-mean:   %f"%Gmean)

def pairwise_pixel_confusion_matrix(im, gt):
    """
    Calculates the pairwise pixel confusion matrix for the label images im and the ground truth gt.
    Returns true-positives, false-positives, false-negatives, and true-negatives as a 2x2 array.
    """
    if im.shape != gt.shape: raise ValueError('Image and Ground Truth not the same shape')

    # Remove all background pixels
    msk = gt != 0
    im,gt = im[msk], gt[msk]
    del msk
    im -= 1; gt -= 1 # shift values so 0 is a valid label
    
    # Count up each pair of im-lbl to gt-lbl
    # This is written in Cython since a pure-Python method is 10-20x slower
    from .__count_pairs import count_pairs
    pairs = count_pairs(im, gt)
    
    # Calculate the confusion matrix
    n_pair = __double_sum(pairs.sum())
    n_pair_gt = __double_sum(pairs.sum(axis=0))
    n_pair_im = __double_sum(pairs.sum(axis=1))
    TP = __double_sum(pairs).sum()
    TN = n_pair + TP - n_pair_gt - n_pair_im
    FP = n_pair_im - TP
    FN = n_pair_gt - TP
    
    # Return the confusion matrix
    from numpy import array
    return array([[TP, FP], [FN, TN]], dtype='uint64')
    
def __double_sum(a):
    """Calculates sum(a*(a-1)/2)"""
    x = a - 1; x *= a; x //= 2
    return x.sum()

if __name__ == "__main__": main()

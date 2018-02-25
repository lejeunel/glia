#!/usr/bin/env python2
"""
Statistics utilities. Measures the Accuracy, F-value, and G-mean used pairwise pixel values.

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
    from .__pairwise_pixel_values import pairwise_pixel_values
    from pysegtools.images.filters.label import RelabelImageStack
    from math import sqrt
    
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
    
    # Make sure we have grayscale images
    ims = RelabelImageStack(ims, per_slice=not threeD)
    gts = RelabelImageStack(ims, per_slice=not threeD)
    
    # Count up confusion matrix
    if threeD:
        im = ims.stack
        gt = gts.stack
        total_im_labels = im.max()
        total_gt_labels = gt.max()
        confusion_matrix = pairwise_pixel_values(im.ravel(), gt.ravel())
    else:
        total_im_labels = 0
        total_gt_labels = 0
        confusion_matrix = [0,0,0,0]
        for im,gt in izip((im.data for im in ims), (gt.data for gt in gts)):
            total_im_labels += im.max()
            total_gt_labels += gt.max()
            TP,FP,FN,TN = pairwise_pixel_values(im.ravel(), gt.ravel())
            confusion_matrix[0] += TP
            confusion_matrix[1] += FP
            confusion_matrix[2] += FN
            confusion_matrix[3] += TN
       
    # Calculate metrics
    TP,FP,FN,TN = confusion_matrix
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    specificity = TN / (TN+FP)
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

if __name__ == "__main__": main()

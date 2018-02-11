#!/usr/bin/env python2
"""
Contour Utilities. Converts label images into contours/polygons.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from .__contours_around_labels import contours_around_labels

def __collinear(a, b, c, thresh=1e-10):
    """Checks if 3 2D points are collinear."""
    x1,y1 = a; x2,y2 = b; x3,y3 = c
    return abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)) < thresh

def simplify_contour(cont, thresh=1e-10):
    """
    Simplifies a contour by removing collinear points. The data must be Nx2. The returned result
    is the contour converted to an array possibly missing some rows. Collinear points are defined
    as any point where the absolute value of the determinant of three consecutive points in the
    contour is less than the given threshold (defaults to 1e-10). Note that currently the
    contours_around_labels function already simplifies the contours as it is processing the data.
    """
    from numpy import asarray, flatnonzero, delete
    cont = asarray(cont)
    if len(cont) < 3: return cont
    x1,x2,x3 = cont[:-2,0], cont[1:-1,0], cont[2:,0]
    y1,y2,y3 = cont[:-2,1], cont[1:-1,1], cont[2:,1]
    collinear = flatnonzero(abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)) < thresh)
    if len(collinear) != 0: cont = delete(cont, collinear+1, 0)
    if len(cont) >= 3 and __collinear(cont[-1], cont[ 0], cont[1], thresh): cont = cont[1:]
    if len(cont) >= 3 and __collinear(cont[-2], cont[-1], cont[0], thresh): cont = cont[:-1]
    return cont

def merge_contour_lists(*args):
    """Merges two or more dictionaries as given by contours_around_labels into a single dictionary."""
    from itertools import islice
    if len(args) == 0: return {}
    d = args[0].copy()
    for a in islice(args, 1, None):
        for k,v in a.iteritems(): d.setdefault(k, []).extend(v)
    return d

def write_points(f, contours, starts=None):
    """
    Outputs the contours as a "points" file with each line in the file listing the object id,
    contour id within the object, and then x, y, and z for each point. This file is directly
    compatible with the IMOD program point2model. The file-like object f must have a write
    method. An optional argument starts is a dictionary of starting values for the contour
    numbers for each object, defaulting to all 1s. This function returns the ending values
    for use to another call to write_points as the starts.
    """
    if starts is None: starts = {}
    for obj_id,conts in contours.iteritems():
        for cont_id,cont in enumerate(conts, starts.setdefault(obj_id, 1)):
            for x,y,z in cont:
                f.write("%d %d %.2f %.2f %.2f\n" % (obj_id, cont_id, x, y, z))
        starts[obj_id] += len(conts)
    return starts

def __contours_around_labels_iter(args):
    z, im = args
    return contours_around_labels(im, z-0.5)

def main():
    from argparse import ArgumentParser, FileType
    from multiprocessing import Pool, cpu_count
    from itertools import imap
    from ._utils import int_check, open_image_stack
    ncpus = cpu_count()
    
    # Setup parser
    parser = ArgumentParser(description="""Converts label images to point files containing contours around each label.

Each line of the output file contains the label number, contour number, and the X, Y, and Z coordinates for each point. They can directly be used by the IMOD program point2model to create a model file.""",
                            fromfile_prefix_chars='@')
    parser.add_argument('image-stack', type=open_image_stack,
                        help='the image stack of labels, this accepts any grayscale image stacks that '+
                        'can be given to `imstack -L` except that it may need to be in quotes')
    parser.add_argument('output.pts', type=FileType('w'),
                        help='the output text file (or - for stdout) to write the contour points to')
    parser.add_argument('-N', dest='nthreads', type=int_check(1, ncpus), default=ncpus,
                        help='number of threads to use, default is number of CPUs')
    
    # Get arguments
    args = parser.parse_args()
    _,ims = getattr(args, 'image-stack')
    out = getattr(args, 'output.pts')
    nthreads = args.nthreads
    if nthreads != 1: imap = Pool(nthreads).imap
    
    # Calculate and write out all of the contours
    offs = {}
    for contours in imap(__contours_around_labels_iter, enumerate(im.data for im in ims)):
        offs = write_points(out, contours, offs)
    out.close()

if __name__ == "__main__": main()


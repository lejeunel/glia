#!/usr/bin/env python2
"""
LDNN testing wrapper.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

def test(model, X):
    """
    Perform LDNN testing for GLIA on a model.
    
    model is a dictionary with:
        
        weights     NxMx(n+1) matrix of weights representing all w_ijk and biases b_ij (in the
                    final row). n is the number of features, N is the number of groups (ORs), and M
                    is the of nodes/discriminants per group (ANDs)
        dropout     (optional) if the model was created using the dropout method, default False
        
        redundant   (optional) list of features to remove from the data, default is none
        
        stats       (optional) stats to use with norm_rows

    X is a Sxn or Sx(n+1) matrix of pixels where S is the number of samples. If it is (n+1)xS then
    the last row will be filled with 1s by this method. Using an (n+1)xS input is faster.
    """
    from numpy import delete
    from chm.ldnn import test #pylint: disable=redefined-outer-name
    from ._utils import norm_rows
    red_feats = model.get('redundant', ())
    if red_feats: X = delete(X, red_feats, 0)
    norm_rows(X, model.get('stats', None))
    return test(model['weights'], X, model.get('dropout', False))

def main():
    from argparse import ArgumentParser
    from numpy import subtract
    from pysegtools.general.json import load as load_json
    from ._utils import load_txt, save_txt
    
    parser = ArgumentParser(description='Runs LDNN-test for a set of features',
        epilog='The `feats` and `preds` files can be CSV, TSV, or space separated value. Additionally if they have an extension of .gz or .bz2 they will be (de)compressed transparently.')
    parser.add_argument('model', help='the file that contains the model')
    parser.add_argument('feats', help='the input file that contains the features')
    parser.add_argument('preds', help='the output file that will contain the predictions')
    parser.add_argument('--flip-preds', action='store_true',
        help='if given will flip the prediction outputs so that 1 becomes 0 and 0 becomes 1')
    args = parser.parse_args()
    
    # Load model
    model = load_json(args.model)
    
    # Load features
    feats = load_txt(args.feats).T  # D-by-N
    
    # Calculate predictions
    preds = test(model, feats).T
    if args.flip_preds: subtract(1, preds, preds)
    
    # Save predictions
    save_txt(args.preds, preds)

if __name__ == "__main__": main()


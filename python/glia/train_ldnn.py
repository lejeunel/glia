#!/usr/bin/env python2
"""
LDNN training wrapper.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

__def_ldnn_params = {'N':15,'M':15,'downsample':1,'kmeans_rep':5,'batchsz':1,'niters':50,'rate':0.025}

def train(X, Y, remove_redundant=True, balance=False, ldnn_params=None):
    from numpy import delete, unique
    from chm.ldnn import learn
    from chm.utils import ensure_binary
    from ._utils import norm_rows

    if ldnn_params is None: ldnn_params = __def_ldnn_params
    
    # Setup the model
    model = { }
    if ldnn_params.get('dropout', False): model['dropout'] = True
    
    # Find and remove redundant features
    if remove_redundant:
        red_feats = ()
        for d in xrange(X.shape[0]):
            if len(unique(X[d, :])) == 1: # TODO: seems like this could be improved greatly
                red_feats = red_feats + (d,)
        if red_feats:
            X = delete(X, red_feats, 0)
            model['redundant'] = red_feats
  
    # Normalize the columns
    model['stats'] = norm_rows(X)
    
    # Binarize the data
    Y = ensure_binary(Y)

    # Perform balancing
    if balance: balance_samples(X, Y)
    
    # Print out some information
    if ldnn_params.get('disp', True):
        n1 = Y.sum()
        n0 = len(Y) - n1
        print('- to + = %d : %d = %s : %s'%((n0,n1,1,n1/n0) if n0 < n1 else (n0,n1,n0/n1,1)))
    
    # Run training
    model['weights'] = learn(X, Y, **ldnn_params)
    
    return model

def balance_samples(X, Y):
    """
    Balances the number of positive and negative samples by duplicating samples from the smaller
    one.
    """
    from numpy import where, concatenate, repeat
    from numpy.random import choice, permutation
    n_pos = Y.sum() # number of positive samples
    n_neg = len(Y) - n_pos # number of negative samples
    if n_pos == n_neg: return X, Y
    
    # Add either n_pos-n_neg negative samples or n_neg-n_pos positive samples
    Y = concatenate((Y, repeat(n_neg>n_pos, abs(n_pos-n_neg))))
    i,(d,r) = (where(~Y),divmod(n_pos, n_neg)) if n_neg < n_pos else where(Y),divmod(n_neg, n_pos)
    i = concatenate((repeat(i, d-1), choice(i, r, False)))
    X = concatenate((X, X.take(i, 0)), 0)
    
    # Shuffle the data so all of the new samples aren't at the end
    ii = permutation(len(Y))
    return X.take(ii, 0), Y.take(ii)

def main():
    from argparse import ArgumentParser
    from numpy import concatenate
    from pysegtools.general.json import load as load_json, save as save_json
    from ._utils import load_txt

    # Setup the argument parser
    parser = ArgumentParser(description='Runs LDNN-train on a set of features',
        epilog='The input files can be space or comma separated values (as long as the extension is .csv) and if they have an extension of .gz or .bz2 they will be decompressed transparently.')
    parser.add_argument('model', help='the file that will contain the model')
    parser.add_argument('-a', '--append', action='store_true', help='append the model data to the given model file')
    parser.add_argument('-f', '--features', help='the input files that contain the features', nargs='+', required=True)
    parser.add_argument('-l', '--labels', help='the input files that contain the labels', nargs='+', required=True)
    parser.add_argument('-b', '--balance', action='store_true', help='if given will balance the positive and negative sample counts by duplicating samples from the smaller one')
    parser.add_argument('-r', '--keep_redundant', action='store_true', help='if given will not run redudant feature detection')
    # TODO: alternative normalization methods that can be selected
    # TODO: add subsampling option
    
    # LDNN training arguments
    parser.add_argument('-N', '--ngroups', dest='N', type=int, help='the number of groups to use in the model', default=15)
    parser.add_argument('-M', '--ndisc-per-grp', dest='M', type=int, help='the number of discriminants per group to use in the model', default=15)
    parser.add_argument('-C', '--cluster-downsample', dest='downsample', type=int,
        help='the amount to downsample the date for clustering to increase the speed of calcalating the initial weights', default=1)
    parser.add_argument('-K', '--kmeans-repeat', type=int, help='the number of times to repeat running k-means clustering to get a better initial weights', default=5)
    parser.add_argument('-B', '--batch-size', type=int, help='the mini-batch size to use to increase the speed of each iteration of gradient descent', default=1)
    parser.add_argument('-n', '--niters', type=int, help='the of iterations of gradient descent to perform', default=50)
    parser.add_argument('-R', '--rate', type=float, help='the learning rate to use during gradient descent', default=0.025)
    # TODO: dropout, momentum, target
    
    # Parse the arguments
    args = parser.parse_args()

    # Load features and labels
    if len(args.features) != len(args.labels):
        parser.error('Number of features and labels must be the same')
    feats = concatenate([load_txt(f) for f in args.features], 0)
    lbls = concatenate([load_txt(f) for f in args.labels])
    if len(feats) != len(lbls):
        parser.error('Number of features and labels must be the same')

    # Train model
    ldnn_params = {'N':args.N,'M':args.M,'downsample':args.downsample,
                   'kmeans_rep':args.kmeans_repeat,'batchsz':args.batch_size,
                   'niters':args.niters,'rate':args.rate}
    model = train(feats.T, lbls, not args.keep_redundant, args.balance, ldnn_params)

    # Save model
    if args.append:
        old = load_json(args.model)
        old.update(model)
        model = old
    save_json(args.model, model)

if __name__ == "__main__": main()


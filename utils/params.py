import os
import configargparse
from os.path import join as pjoin


def get_params(path='.'):
    """ Builds default configuration
    """
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[pjoin(path, 'default.yaml')])

    p.add('-v', help='verbose', action='store_true')

    p.add('--slic-segments', type=int)
    p.add('--slic-compactness', type=float)

    p.add('--n-ori', type=int)
    p.add('--k-mean-num', type=int)
    p.add('--sigma-sm', type=float)
    p.add('--sigma-lg', type=float)

    p.add('--daisy-step', type=int)
    p.add('--daisy-radius', type=int)
    p.add('--daisy-rings', type=int)
    p.add('--daisy-orientations', type=int)
    p.add('--daisy-histograms', type=int)

    p.add('--codebook-size', type=int)
    p.add('--kmeans-thresh', type=float)
    p.add('--kmeans-n-samples', type=int)

    p.add('--hist-bins-color', type=int)
    p.add('--hist-bins-daisy', type=int)
    p.add('--hist-bins-texture', type=int)
    p.add('--initial-saliency', type=float)
    p.add('--saliency-bias', type=float)
    p.add('--normalize-area', type=bool, default=True)
    p.add('--use-log-shape', type=bool, default=False)

    p.add('--cuda', default=False, action='store_true')

    p.add('--n-trees', type=int)
    p.add('--sample-size-ratio', type=float)
    p.add('--balance', type=bool, default=True)
    p.add('--n-classifiers', type=int)

    return p

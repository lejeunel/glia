from glia import libglia
from feat_extr import get_features
from hed.run import Network
import torch
import params
from pascal_voc_loader import pascalVOCLoader
from loader import Loader
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
import numpy as np
import os
from os.path import join as pjoin
from skimage.feature import daisy
from skimage import transform, segmentation
from scipy import cluster


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    hed_network = Network()
    hed_network.load_pretrained().to(device).eval()

    assert (cfg.mode == 'pascal' or cfg.mode == 'medical'), 'mode must be pascal or medical'

    if(not os.path.exists(cfg.out_path)):
        os.makedirs(cfg.out_path)

    if(cfg.mode == 'pascal'):
        loader = pascalVOCLoader(cfg.in_path)
    else:
        loader = Loader(root_path=cfg.in_path,
                        truth_type='hand')

    hmt = libglia.hmt.create()
    hmt.config(3, 255, 0, 0.7, True)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    hed_network = Network()
    hed_network.load_pretrained().to(device).eval()

    hmt = libglia.hmt.create()
    for i, sample in enumerate(loader):
        print('{}/{}'.format(i+1, len(loader)))

        img = sample['image']

        feats = get_features(img, cfg, hed_network, hmt)
        X = feats['bc_feats']
        Y = hmt.bc_label_ri(feats['order'],
                                feats['labels'],
                                sample['label/segmentation'],
                                True,
                                False,
                                0,
                                False,
                                1.0)
                                

if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path', required=True)
    p.add('--out-path', required=True)
    p.add('--mode', required=True)

    cfg = p.parse_args()

    main(cfg)





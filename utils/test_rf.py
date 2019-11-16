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

    if (not os.path.exists(cfg.out_path)):
        os.makedirs(cfg.out_path)

    hmt = libglia.hmt.create()

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    hed_network = Network()
    hed_network.load_pretrained().to(device).eval()

    hmt = libglia.hmt.create()
    hmt.config(3, cfg.n_trees, 0, cfg.sample_size_ratio, cfg.balance)

    # for each clique, compute features for the boundary classifier
    bc_feats_fn = lambda order, saliencies, labels, img_lab, img_hsv, daisy_descs, contours: hmt.bc_feat(
        order, saliencies, labels, [img_lab, img_hsv] + daisy_descs, contours,
        [cfg.hist_bins_color, cfg.hist_bins_color] + 3 * [cfg.hist_bins_daisy],
        [0., -127., -128., 0., 0., 0., 0., 0., 0.], [
            100., 128., 127., 1., 1., 1., 256., 256., 256.
        ], cfg.initial_saliency, cfg.saliency_bias, [0.2, 0.5, 0.8], cfg.
        normalize_area, cfg.use_log_shape)

    # for all frames, compute features and labels
    phases = ['train', 'test']
    feats = {k: [] for k in phases}
    for phase in phases:
        if(phase == 'train'):
            path = cfg.in_path_train
            frames = cfg.train_frames
        else:
            path = cfg.in_path_test
            frames = cfg.test_frames

        loader = Loader(root_path=path, truth_type='hand')
        samples = [loader[f] for f in frames]
        for i, sample in enumerate(samples):
            print('[{}]: extracting features {}/{}'.format(phase, i + 1, len(samples)))
            img = sample['image']

            feats[phase].append(get_features(img, cfg, hed_network))

    # aggregate training features

    # train boundary classifier with aggregated features
    for t in range(cfg.n_classifiers):
        print('training classifier {}/{}'.format(i + 1, cfg.n_classifiers))
        for i, s in enumerate(feats['train']):
            print('{}/{}'.format(i + 1, len(feats['train'])))

            sample = loader[f]
            img = sample['image']

            feats = get_features(img, cfg, hed_network)
            if (t == 0):
                # This is the merge based on boundary probabilities (first tree in the ensemble)
                order, saliencies = hmt.merge_order_pb(feats['labels'],
                                                       feats['contours'], 1)
            else:
                order, saliencies = hmt.merge_order_bc()

            X = bc_feats_fn(order, saliencies, feats['labels'], feats['img_lab'],
                            feats['img_hsv'], feats['daisy_descs'], feats['contours'])
            X_list.append(X)

            # for each merge, compute label based on groundtruth
            # -1 is for "merge"
            # 1 is for "no merge"
            Y = hmt.bc_label_ri(feats['order'], feats['labels'],
                                sample['label/segmentation'], True, 0, False,
                                False, 1.0)
            Y_list.append(Y)

        hmt.train_rf(np.concatenate(X_list, axis=0),
                     np.concatenate(Y_list))


if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path-train', required=True)
    p.add('--train-frames', action='append', type=int, required=True)
    p.add('--out-path', required=True)
    p.add('--in-path-test', required=True)
    p.add('--test-frames', action='append', type=int, required=True)

    cfg = p.parse_args()

    main(cfg)

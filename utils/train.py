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
import pickle


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
        order, saliencies, labels, img_lab + img_hsv + daisy_descs, contours,
        6 * [cfg.hist_bins_color] + 3 * [cfg.hist_bins_daisy],
        [0., -127., -128., 0., 0., 0., 0., 0., 0.], [
            100., 128., 127., 1., 1., 1., 256., 256., 256.
        ], cfg.initial_saliency, cfg.saliency_bias, [0.2, 0.5, 0.8], cfg.
        normalize_area, cfg.use_log_shape)

    # generate merge order using boundary classifier
    merge_order_bc_fn = lambda labels, img_lab, img_hsv, daisy_descs, contours, thr: hmt.merge_order_bc(
        labels, img_lab + img_hsv + daisy_descs, contours,
        6 * [cfg.hist_bins_color] + 3 * [cfg.hist_bins_daisy],
        [0., -127., -128., 0., 0., 0., 0., 0., 0.], [
            100., 128., 127., 1., 1., 1., 256., 256., 256.
        ], cfg.use_log_shape, thr)

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
            print('[{}]: extracting image features {}/{}'.format(phase, i + 1, len(samples)))
            img = sample['image']

            feats[phase].append(get_features(img, cfg, hed_network))

    X_list = []
    Y_list = []
    # train boundary classifier with aggregated features
    for t in range(cfg.n_classifiers):
        print('training classifier {}/{}'.format(i + 1, cfg.n_classifiers))
        for i, s in enumerate(feats['train']):
            print('{}/{}'.format(i + 1, len(feats['train'])))

            truth = loader[i]['label/segmentation']
            img = sample['image']

            if (t == 0):
                # This is the merge based on boundary probabilities (first tree in the ensemble)
                order, saliencies = hmt.merge_order_pb(s['labels'],
                                                       s['contours'], 1)
            else:
                order, saliencies = merge_order_bc_fn(s['labels'],
                                                      s['img_lab'],
                                                      s['img_hsv'],
                                                      s['daisy_descs'],
                                                      s['contours'],
                                                      hmt.get_threshold())

            X = bc_feats_fn(order, saliencies, s['labels'], s['img_lab'],
                            s['img_hsv'], s['daisy_descs'], s['contours'])
            X_list.append(X)

            # for each merge, compute label based on groundtruth
            # -1 is for "merge"
            # 1 is for "no merge"
            Y = hmt.bc_label_ri(order, s['labels'],
                                truth, True, 0, False,
                                False, 1.0)
            Y_list.append(Y)

        hmt.train_rf(np.concatenate(X_list, axis=0),
                     np.concatenate(Y_list))

    if(not os.path.exists(cfg.out_path)):
        os.makedirs(cfg.out_path)

    path = pjoin(cfg.out_path, 'models.p')
    models = hmt.get_models()
    print('Saving models to {}'.format(path))
    pickle.dump(models, open(path, 'wb'))


if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path-train', required=True)
    p.add('--train-frames', action='append', type=int, required=True)
    p.add('--out-path', required=True)
    p.add('--in-path-test', required=True)
    p.add('--test-frames', action='append', type=int, required=True)

    cfg = p.parse_args()

    main(cfg)

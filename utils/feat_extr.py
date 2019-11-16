from glia import libglia
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


def encode(feats, n_samples, cb_size, thresh):
    orig_shape = feats.shape[:2]
    descs = feats.reshape(-1, feats.shape[-1])
    feats_for_cluster = cluster.vq.whiten(descs)
    feats_for_cluster = feats_for_cluster[np.random.choice(
        feats_for_cluster.shape[0], n_samples, replace=False), :]
    cb, _ = cluster.vq.kmeans(feats_for_cluster, cb_size, thresh=thresh)
    feats_codes, _ = cluster.vq.vq(descs, cb)
    feats_codes = feats_codes.reshape(orig_shape[0], orig_shape[1])

    return feats_codes


def get_features(img, cfg, hed_network):
    device = torch.device('cuda' if cfg.cuda else 'cpu')
    img_tnsr = torch.from_numpy(np.rollaxis(img / 255, -1,
                                            0)[None, ...]).to(device)
    print('Estimating HED contours')
    contours = hed_network(img_tnsr.float())
    contours = contours.squeeze().detach().cpu().numpy()

    labels = segmentation.slic(img,
                               n_segments=cfg.slic_segments,
                               compactness=cfg.slic_compactness)
    print('num. of labels: {}'.format(np.unique(labels).size))

    img_lab = color.rgb2lab(img)
    img_hsv = color.rgb2hsv(img)

    my_daisy = lambda im: daisy(np.pad(im, cfg.daisy_radius),
                                step=cfg.daisy_step,
                                radius=cfg.daisy_radius,
                                rings=cfg.daisy_rings,
                                histograms=cfg.daisy_histograms,
                                orientations=cfg.daisy_orientations)
    my_encode = lambda f: encode(f, cfg.kmeans_n_samples, cfg.codebook_size,
                                 cfg.kmeans_thresh)

    print('Coding texture descriptors to {} words'.format(cfg.codebook_size))
    daisy_descs = [
        my_encode(my_daisy(img.mean(axis=-1))),
        my_encode(my_daisy(img_lab[..., 1])),
        my_encode(my_daisy(img_lab[..., 2]))
    ]

    orig_shape = img.shape
    resized = False

    data = {
        'img': img,
        'img_hsv': img_hsv,
        'img_lab': img_lab,
        'daisy_descs': daisy_descs,
        'contours': contours,
        'labels': labels,
    }

    return data


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    hed_network = Network()
    hed_network.load_pretrained().to(device).eval()

    assert (cfg.mode == 'pascal'
            or cfg.mode == 'medical'), 'mode must be pascal or medical'

    if (not os.path.exists(cfg.out_path)):
        os.makedirs(cfg.out_path)

    if (cfg.mode == 'pascal'):
        loader = pascalVOCLoader(cfg.in_path)
    else:
        loader = Loader(root_path=cfg.in_path, truth_type='hand')

    for i, sample in enumerate(loader):
        print('{}/{}'.format(i + 1, len(loader)))

        path = pjoin(
            cfg.out_path,
            '{}.npz'.format(os.path.splitext(sample['frame_name'])[0]))
        if (not os.path.exists(path)):
            img = sample['image']

            feats = get_features(img, cfg, hed_network)

            print('writing features to {}'.format(path))

            np.savez(path, **feats)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path', required=True)
    p.add('--out-path', required=True)
    p.add('--mode', required=True)

    cfg = p.parse_args()

    main(cfg)

import os
from os.path import join as pjoin
from skimage import io, color
import glob
import numpy as np
import matplotlib.pyplot as plt

def imread(path, scale=True):
    im = io.imread(path)

    if(im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (scale):
        im = im / 255

    if(len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im


class Loader:
    def __init__(self,
                 root_path=None,
                 feats_path=None,
                 truth_type=None):
        """

        """

        self.root_path = root_path
        self.feats_path = feats_path
        self.truth_type = truth_type

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(sorted(glob.glob(pjoin(root_path,
                                           'input-frames',
                                                     e))))
        truth_paths = []
        for e in exts:
            truth_paths.extend(sorted(glob.glob(pjoin(root_path,
                                        'ground_truth-frames',
                                                    e))))
        self.truth_paths = truth_paths
        self.img_paths = img_paths

        self.truths = [
            io.imread(f).astype('bool') for f in self.truth_paths
        ]
        self.truths = [t if(len(t.shape) < 3) else t[..., 0]
                        for t in self.truths]

        self.imgs = []
        for f in self.img_paths:
            im_ = io.imread(f)
            if(im_.strides[1] > 3):
                im_ = (255 * color.rgba2rgb(im_)).astype(np.uint8)
            if(len(im_.shape) < 3):
                im_ = np.repeat(im_[..., None], 3, -1)
            self.imgs.append(im_)

        if(self.feats_path is not None):
            self.feats_paths = sorted(glob.glob(pjoin(self.feats_path, '*.npz')))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        truth = self.truths[idx]
        im = self.imgs[idx]

        feats = np.load(self.feats_paths[idx])

        return {'image': self.imgs[idx],
                'frame_idx': idx,
                'feats': feats,
                'frame_name': os.path.split(self.img_paths[idx])[-1],
                'label/segmentation': self.truths[idx]}

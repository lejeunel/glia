from glia import utils as utls
import matplotlib.pyplot as plt
from glia import libglia
import os
from skimage import io, segmentation, color

path_ = os.path.join('/home',
                     'krakapwa',
                     'Documents',
                     'software',
                     'glia_my')

path_im_in = os.path.join(path_, 'frame_0482.png')
path_gpb_in = os.path.join(path_, 'frame_0482_gPb.png')
path_ucm_in = os.path.join(path_, 'frame_0482_ucm.png')
path_gt_in = os.path.join(path_, 'frame_0482_gt.png')
gpb = (io.imread(path_gpb_in).astype(float))/255
ucm = (io.imread(path_ucm_in).astype(float))/255

bc_truth = utls.truth_to_bc_truth(io.imread(path_gt_in))

img = io.imread(path_im_in)
img_hsv = color.rgb2hsv(img)
img_lab = color.rgb2lab(img)
img_dict = {'h': img_hsv[..., 0],
            's': img_hsv[..., 1],
            'v': img_hsv[..., 2],
            'l': img_lab[..., 0],
            'a': img_lab[..., 1],
            'b': img_lab[..., 2],
            'gray': np.mean(img, axis=-1)}

# Compute daisy on gray, A and B channels
daisy_feats = utls.comp_daisy_feats([img_dict['gray'],
                                img_dict['a'],
                                img_dict['b']])

print('Computing watershed')
labels = libglia.watershed(img.copy(),
                           1,
                           True)

print('Num superpixels before merge {} '.format(np.unique(labels).size))

merged_labels = libglia.pre_merge(labels.copy(),
                                  gpb,
                                  np.array([]),
                                  [50, 400],
                                  0.5,
                                  True)

print('Num superpixels after merge {} '.format(np.unique(merged_labels).size))

order, saliencies = libglia.merge_order_pb(merged_labels.copy(),
                                           gpb,
                                           np.array([]),
                                           2)


data = {'imgs': img_dict,
        'daisy': daisy_feats,
        'labels': labels,
        'merged_labels': merged_labels,
        'order': order,
        'saliencies': saliencies,
        'gpb': gpb,
        'ucm': ucm}

np.savez(os.path.join(path_, 'data.npz'), **data)

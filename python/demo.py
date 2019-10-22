from skimage import io, segmentation
import matplotlib.pyplot as plt
from glia import libglia
import numpy as np
import os
import networkx as nx
from sklearn.ensemble import RandomForestClassifier

path_ = os.path.join('/home',
                     'krakapwa',
                     'Documents',
                     'software',
                     'glia_my')

np_file = np.load(os.path.join(path_, 'data.npz'))

truth = io.imread(os.path.join(path_, 'frame_0482_gt.png'))

img_dict = np_file['imgs'][()]
daisy_feats = np_file['daisy']
labels = np_file['labels']
merged_labels = np_file['merged_labels']
order = np_file['order']
saliencies = np_file['saliencies']
gpb = np_file['gpb']
ucm = np_file['ucm']

imgs_for_feats = [img_dict[k]
                  for k in img_dict.keys()
                  if(k!='gray')]
imgs_for_feats += [daisy_feats[i, ...] for i in range(daisy_feats.shape[0])]

feats = libglia.bc_feat(list(order), #Merge list orders
                saliencies, #Saliency values
                [merged_labels],
                [img_dict[k] for k in img_dict.keys() ], #Image
                [gpb, ucm], #global Probability Boundary map
                np.array([]), #mask array (unused)
                3*[16], #Num. of histogram bins per channel
                3*[0.], #Lower bounds of histograms
                3*[1.],#Upper bounds of histograms
                1., #initial saliency value
                1., #bias of saliency
                [0.2, 0.5, 0.8], #boundary shape thresholds
                False, #normalize size and lengths?
                True) # use log of shapes as features?

y = libglia.bc_label_ri(list(order),
                        merged_labels,
                        truth,
                        np.array([]),
                        True,
                        0,
                        False,
                        False,
                        1.0)

cats = categorize_cliques(order, merged_labels)
Y = [y[cats == c] for c in range(1, 4)]
X = [feats[cats == c, :] for c in range(1, 4)]

for i in range(0, 3):
    if(Y[i] ):

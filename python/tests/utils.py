import numpy as np
from skimage.feature import daisy
from skimage import io, segmentation
from skimage import color
from skimage import measure
from scipy.cluster import vq

def categorize_cliques(order, labels):

    sizes = np.bincount(labels.ravel())
    median = np.median(sizes)
    cats = []

    g = order_list_to_graph(order, labels)

    for j in range(order.shape[1]):
        if(np.max((g.node[order[0,j]]['size'],
                   g.node[order[1,j]]['size'])) < median):
           cats.append(1)
        elif(np.min((g.node[order[0,j]]['size'],
                     g.node[order[1,j]]['size']) < median) &
              np.max((g.node[order[0,j]]['size'],
                      g.node[order[1,j]]['size']) >= median)):
            cats.append(2)
        else:
           cats.append(3)

    # Eliminate label 0 (non-existant)
    return np.asarray(cats)

def order_list_to_graph(order, labels):

    g = nx.DiGraph()
    sizes = np.bincount(labels.ravel())

    for l in np.unique(labels):
        g.add_node(l, size=sizes[l])

    for j in range(order.shape[1]):
        s_ = g.node[order[0, j]]['size'] + g.node[order[1, j]]['size']
        g.add_node(order[2, j], size=s_)
        g.add_edge(order[2, j], order[1,j])
        g.add_edge(order[2, j], order[0,j])

    return g

def make_mean_regions(img, labels):

    img_out = np.zeros(img.shape, dtype=np.uint8)

    for l in np.unique(labels):
        mean_color = np.round(np.mean(img[labels == l, :],
                                      axis = 0)).astype(np.uint8)
        img_out[labels==l, :] = mean_color

    return img_out


def make_mean_regions(img, labels):

    img_out = np.zeros(img.shape, dtype=np.uint8)
    for l in np.unique(labels):
        mean_color = np.round(np.mean(img[labels == l, :], axis = 0)).astype(np.uint8)
        img_out[labels==l, :] = mean_color

    return img_out

def comp_daisy_feats(im_list):

    k_means_factor = 0.05
    k_means_iters = 5

    rings = 2
    histograms = 6
    orientations = 8
    radius = 58
    step = 1
    orig_size = im_list[0].shape
    dim = (rings * histograms + 1) * orientations
    pad_width = (int(np.ceil((radius)/step)),
                 int(np.ceil((radius)/step)))

    im_list_pad = [np.pad(im, pad_width, mode='constant') for im in im_list]

    descs_daisy = [daisy(im, step=step, radius=radius, rings=rings,
                         histograms=histograms,
                         orientations=orientations).reshape((-1, dim))
                   for im in im_list_pad]

    # Pick k_means_factor only
    descs_daisy_downsamp = [d[np.random.choice(np.arange(d.shape[0]),
                                               int(d.shape[0]*k_means_factor)),:]
                            for d in descs_daisy]

    codebooks = [vq.kmeans(vq.whiten(d), 256, iter=k_means_iters)[0]
                 for d in descs_daisy_downsamp]

    codes = [vq.vq(vq.whiten(d), cb)[0] for d,cb in zip(descs_daisy, codebooks)]
    codes = [c.reshape(orig_size) for c in codes]

    return codes

def truth_to_bc_truth(truth):
    # Compute ground truth for boundary classifier from segmentation truth

    bc_truth = []
    values = np.unique(truth)
    for v in values:
        c_ = measure.find_contours(truth, v)
        bc_truth.append(c_)

    return bc_truth

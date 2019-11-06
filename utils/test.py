import cPickle
from glia import libglia
import utils as utls
import params
from loader import Loader
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from os.path import join as pjoin
from tqdm import tqdm

def main(cfg):

    loader = Loader(root_path=cfg.in_path,
                    feats_path=pjoin(cfg.run_path, 'features'),
                    truth_type='hand')

    # construct samples and labels for RF
    print('Constructing samples for RF')
    clique_samples = [{'feat': None, 'cat': None, 'label': None} for _ in range(len(loader))]

    bar = tqdm.tqdm(total=len(loader))
    for i, s in enumerate(loader):
        bar.update(1)
        feat = s['feats']
        clique_cats = utls.categorize_cliques(feat['order'], feat['merged_labels'])
        clique_samples[i]['feat'] = feat['bc_feats']
        clique_samples[i]['cat'] = clique_cats
    bar.close()

    # concatenate
    clique_feats = np.concatenate([c['feat'] for c in clique_samples], axis=0)
    clique_cats = np.concatenate([c['cat'] for c in clique_samples], axis=0)
    clique_labels = np.concatenate([c['label'] for c in clique_samples], axis=0)

    classifiers = {i: None for i in range(3)}

    # for each category of clique, train RF
    for i in range(3):
        X = clique_feats[clique_cats == i+1, :]
        Y = clique_labels[clique_cats == i+1]
        if(X.size > 0):
            rf = RandomForestClassifier(class_weight='balanced', n_estimators=cfg.n_trees)
            rf.fit(X, Y)
            classifiers[i] = rf

    # save the classifiers
    path = pjoin(cfg.run_path, 'classifiers.p')
    print('saving classifiers to {}'.format(path))
    with open(path, 'wb') as fid:
        cPickle.dump(classifiers, fid)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path', required=True)
    p.add('--run-path', required=True)
    p.add('--path-classifier', type=int, nargs='+', required=True)

    cfg = p.parse_args()

    main(cfg)




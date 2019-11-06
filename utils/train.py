from glia import libglia
import utils as utls
import params
from loader import Loader
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from os.path import join as pjoin
import copy


def main(cfg):

    loader = Loader(root_path=cfg.in_path,
                    feats_path=pjoin(cfg.run_path, 'features'),
                    truth_type='hand')

    samples = [s for i, s in enumerate(loader) if i in cfg.train_frames]

    # construct samples and labels for RF
    clique_samples = [{'feat': None, 'cat': None, 'label': None} for _ in range(len(samples))]
    for i, s in enumerate(samples):
        feat = s['feats']
        clique_truths = libglia.bc_label_ri(feat['order'].tolist(),
                                            feat['merged_labels'],
                                            s['label/segmentation'],
                                            np.array([]),
                                            False,
                                            0,
                                            False,
                                            False,
                                            1.0)
        clique_cats = utls.categorize_cliques(feat['order'], feat['merged_labels'])
        clique_samples[i]['feat'] = feat['bc_feats']
        clique_samples[i]['cat'] = clique_cats
        clique_samples[i]['label'] = clique_truths

    # concatenate
    clique_feats = np.concatenate([c['feat'] for c in clique_samples], axis=0)
    clique_cats = np.concatenate([c['cat'] for c in clique_samples], axis=0)
    clique_labels = np.concatenate([c['label'] for c in clique_samples], axis=0)

    classifiers = [{i: None for i in range(3)} for _ in range(cfg.n_classifiers)]
    X = {i: None for i in range(3)}
    Y = {i: None for i in range(3)}

    for t in range(cfg.n_classifiers):
        # for each category of clique, train RF
        for i in range(3):
            if(t == 0):
                X[i] = clique_feats[clique_cats == i+1, :]
                Y[i] = clique_labels[clique_cats == i+1]
            else:
                Y[i] = rf.fit(X[i])
            if(X[i].size > 0):
                import pdb; pdb.set_trace() ## DEBUG ##
                rf = libglia.train_rf(X[i], Y[i], cfg.n_trees_rf, 0, 0.7, 0, True)
                rf.fit(X[i], Y[i])
                classifiers[t][i] = copy.deepcopy(rf)

    # save the classifiers
    path = pjoin(cfg.run_path, 'classifiers.p')
    print('saving classifiers to {}'.format(path))
    joblib.dump(classifiers, classifiers)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path', required=True)
    p.add('--train-frames', type=int, nargs='+', required=True)
    p.add('--run-path', required=True)
    p.add('--n-trees-rf', type=int, default=255)
    p.add('--n-classifiers', type=int, default=10)

    cfg = p.parse_args()

    main(cfg)




import params, feat_extr
from glia import libglia
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-root-path', required=True)
    p.add('--out-root-path', required=True)
    p.add('--mode', required=True)
    p.add('--sets', nargs='+', required=True)

    cfg = p.parse_args()
    sets = ['Dataset{}'.format(s) for s in cfg.sets]

    for set_ in sets:
        cfg.in_path = pjoin(cfg.in_root_path, set_)
        cfg.out_path = pjoin(cfg.out_root_path, set_)

        feat_extr.main(cfg)

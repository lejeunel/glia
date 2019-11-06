from skimage import segmentation
import numpy as np
import matplotlib.pyplot as plt


path = '/home/ubelix/runs/hmt/Dataset00/features/frame_0400.npz'
# path = '/home/ubelix/runs/hmt/Dataset30/features/frame_0090.npz'
feat = np.load(path)
pb = (feat['contours'] * 255).astype(np.uint8)
img = feat['img']
img_lab = feat['img_lab']
# pb_sp = segmentation.watershed(pb, compactness=1000.)
# pb_sp = segmentation.slic(pb, compactness=1000., n_segments=1000)
sp = segmentation.slic(img, compactness=20., n_segments=1200, max_iter=200)
# pb_sp = segmentation.felzenszwalb(pb, sigma=0.1)
cont_sp = segmentation.find_boundaries(sp, mode='thick')
img[cont_sp, :] = (255, 0, 0)

plt.imshow(img);plt.show()

from skimage import io, segmentation
import matplotlib.pyplot as plt
import libglia
import numpy as np
import os


path_ = os.path.join('/home',
                     'krakapwa',
                     'Documents',
                     'software',
                     'glia_my')

path_im_in = os.path.join(path_, 'frame_0482.png')
path_gpb_in = os.path.join(path_, 'frame_0482_gPb.png')
gpb = (io.imread(path_gpb_in).astype(float))/255

img = io.imread(path_im_in)

# Test conversions
path_im_out_itk_rgb = os.path.join(path_, 'frame_0482_itk_rgb.png')
path_im_out_np_rgb = os.path.join(path_, 'frame_0482_np_rgb.png')
path_im_out_itk_real = os.path.join(path_, 'frame_0482_itk_real.png')
path_im_out_np_real = os.path.join(path_, 'frame_0482_np_real.png')
libglia.test_conversion(img,
                        path_im_in,
                        path_im_out_itk_rgb,
                        path_im_out_itk_real,
                        path_im_out_np_rgb,
                        path_im_out_np_real)

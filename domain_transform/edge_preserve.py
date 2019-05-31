from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from get_data import get_filename, get_batch
#from skimage.restoration import denoise_bilateral
from scipy.ndimage import filters

from multiprocessing import Pool


#def bila_img(img):
#    temp = denoise_bilateral(img, sigma_color=0.1, sigma_spatial=15, multichannel=True)
#    return temp / np.max(temp) * 255.0

def bilateral_filter(old_path):
    print(old_path)
    out_path = old_path.replace("IJCAI_fgsm_output", "holdEdge_output")
    #out_path = old_path
    out_path_dir = os.path.split(out_path)[0] + "/"
    if not os.path.exists(out_path_dir):
        os.makedirs(out_path_dir)

    input_img = Image.open(old_path)
    input_img = np.array(input_img)
    output_img = np.zeros(input_img.shape, np.float32)
    output_img = cv2.edgePreservingFilter(input_img, flags=1, sigma_s=30, sigma_r=0.6)
    Image.fromarray(output_img.astype(np.uint8)).save(out_path)

train, _ = get_filename('./fgsm_val_list.txt')
p = Pool(8)
p.map(bilateral_filter, train)
p.close()
p.join()



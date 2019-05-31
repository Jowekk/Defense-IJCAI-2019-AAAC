from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from get_data import get_filename, get_batch
from scipy.ndimage import filters

from multiprocessing import Pool


def filter_img(img):
    gauss = np.zeros(img1.shape, img1.dtype)
    for j in range(3):
        gauss[:, :, j] = filters.gaussian_filter(img1[:, :, j], 10)
    return gauss

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
    output_img = filter_img(input_img)
    Image.fromarray(output_img.astype(np.uint8)).save(out_path)

train, _ = get_filename('./fgsm_val_list.txt')
p = Pool(8)
p.map(bilateral_filter, train)
p.close()
p.join()



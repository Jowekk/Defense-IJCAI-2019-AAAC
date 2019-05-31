#import cv2
import numpy as np
import PIL.Image
import scipy.misc
from scipy.ndimage import filters
#from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

#----------  gauss filter ---------------
def gauss_filter(A):
    output_imgs = np.zeros((A.shape), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(3):
            output_imgs[i, :, :, j] = filters.gaussian_filter(A[i, :, :, j], 5)
    return output_imgs

#------------ bilatera filter ---------------
'''
def bilateralFilter(A):
    for i in range(A.shape[0]):
        temp = denoise_bilateral(A[i,:,:,:], sigma_color=0.1, sigma_spatial=15, multichannel=True)
        temp = temp / np.max(temp) * 255.0
        A[i,:,:,:] = temp
    return A

def denoise_tv_chambolle(A):
    for i in range(A.shape[0]):
        temp = denoise_tv_chambolle(A[i,:,:,:], weight=0.3, multichannel=True)
        temp = temp / np.max(temp) * 255.0
        A[i,:,:,:] = temp
    return A
'''
#def bilateralFilter(A):
#    for i in range(A.shape[0]):
#        tmp = A[i,:,:,:].astype(np.uint8)
#        tmp = cv2.bilateralFilter(tmp,10,140,140)
#        A[i,:,:,:] = tmp.astype(np.float32)

#------------fft filter ---------------

def convert_2d(r):
    r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_ext[i][j] = r[i][j]

    r_ext_fu = np.fft.fft2(r_ext)
    r_ext_fu = np.fft.fftshift(r_ext_fu)


    d0 = 100

    n = 2

    center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
    h = np.empty(r_ext_fu.shape)

    for u in range(h.shape[0]):
        for v in range(h.shape[1]):
            duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
            h[u][v] = 1 / ((1 + (duv / d0)) ** (2*n))

    s_ext_fu = r_ext_fu * h
    s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
    s_ext = np.abs(s_ext)
    s = s_ext[0:r.shape[0], 0:r.shape[1]]

    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i][j] = min(max(s[i][j], 0), 255)

    return s.astype(np.uint8)

def convert_3d(r):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s

def fft_filter(img_matrix):
    ans = np.zeros(img_matrix.shape)
    for i in range(img_matrix.shape[0]):
        ans[i,:,:,:] = convert_3d(img_matrix[i,:,:,:])
    return ans

#  --------------- tensorflow gaussfilter -------------
def gaussian_kernel(size, mean, std):

    d = tf.contrib.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def tf_gauss_filter(image):
    gauss_kernel = gaussian_kernel(5, 0.0, 5.0)

    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    gauss_kernel = tf.tile(gauss_kernel, multiples=[1, 1, 3, 3])

    after_filter = tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    return after_filter

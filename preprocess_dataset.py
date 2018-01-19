#########################################################################################
####################  Script to prepare dataset (pre-processing) ########################
#########################################################################################

# import libs
import cv2
import numpy as np
from matplotlib import pyplot as plt


# high-pass filter kernel
kernel = np.array(
    [[-1,  2,  -2,  2, -1],
     [2, -6,   8, -6,  2],
     [-2,  8, -12,  8, -2],
     [2, -6,   8, -6,  2],
     [-1,  2,  -2,  2, -1]])

# input dir
input_dir = 'cover_128/'

# loop images
for x in xrange(1, 1700):
	# read img from the input dir
    img = cv2.imread(input_dir + str(x) + '.jpg')
    # resize img
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    # filter the img
    dst = cv2.filter2D(img, -1, kernel)
    # save image in the output dir
    cv2.imwrite('dataset/test/cover/' + str(x) + '.jpg', dst)
    print(x)

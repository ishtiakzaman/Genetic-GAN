import os.path
from scipy import ndimage
import numpy as np
from skimage.io import imsave
import os

pair_file = open('final_good_pair.txt', 'r')
f_file = open('final_good_female.txt', 'r')
m_file = open('final_good_male.txt', 'r')

img_root = '../img_align_celeba/high/'
offspring_root = 'test_good_dump/mean/'

for line in pair_file:
    tokens = line.strip().split(' ')
    n_col = 12
    img_row = 218
    img_col = 178
    image = np.zeros((img_row, img_col*n_col, 3), dtype=np.uint8)
    img_read = ndimage.imread(img_root + tokens[0] + '.jpg', mode='RGB')
    image[0:img_row, img_col*0:img_col*1, :] = img_read
    img_read = ndimage.imread(img_root + tokens[1] + '.jpg', mode='RGB')
    image[0:img_row, img_col*1:img_col*2, :] = img_read
    for ci in xrange(5):
        img_name = tokens[0] + '_' + tokens[1] + '_f_' + str(ci) + '.jpg'
        img_read = ndimage.imread(offspring_root + img_name, mode='RGB')
        image[0:img_row, img_col*(2+ci):img_col*(3+ci), :] = img_read
    for ci in xrange(5):
        img_name = tokens[0] + '_' + tokens[1] + '_m_' + str(ci) + '.jpg'
        img_read = ndimage.imread(offspring_root + img_name, mode='RGB')
        image[0:img_row, img_col*(7+ci):img_col*(8+ci), :] = img_read

    imsave('final_good_ones/pair_' + tokens[0] + '_' + tokens[1] + '.jpg', image)

for line in f_file:
    tokens = line.strip().split(' ')
    n_col = 3
    img_row = 218
    img_col = 178
    image = np.zeros((img_row, img_col*n_col, 3), dtype=np.uint8)
    img_read = ndimage.imread(img_root + tokens[0] + '.jpg', mode='RGB')
    image[0:img_row, img_col*0:img_col*1, :] = img_read
    img_name = tokens[0] + '_' + tokens[1] + '_f_0.jpg'
    img_read = ndimage.imread(offspring_root + img_name, mode='RGB')
    image[0:img_row, img_col*1:img_col*2, :] = img_read
    img_name = tokens[0] + '_' + tokens[1] + '_m_0.jpg'
    img_read = ndimage.imread(offspring_root + img_name, mode='RGB')
    image[0:img_row, img_col*2:img_col*3, :] = img_read

    imsave('final_good_ones/f_' + tokens[0] + '.jpg', image)

for line in m_file:
    tokens = line.strip().split(' ')
    n_col = 3
    img_row = 218
    img_col = 178
    image = np.zeros((img_row, img_col*n_col, 3), dtype=np.uint8)
    img_read = ndimage.imread(img_root + tokens[1] + '.jpg', mode='RGB')
    image[0:img_row, img_col*0:img_col*1, :] = img_read
    img_name = tokens[0] + '_' + tokens[1] + '_f_4.jpg'
    img_read = ndimage.imread(offspring_root + img_name, mode='RGB')
    image[0:img_row, img_col*1:img_col*2, :] = img_read
    img_name = tokens[0] + '_' + tokens[1] + '_m_4.jpg'
    img_read = ndimage.imread(offspring_root + img_name, mode='RGB')
    image[0:img_row, img_col*2:img_col*3, :] = img_read

    imsave('final_good_ones/m_' + tokens[0] + '.jpg', image)

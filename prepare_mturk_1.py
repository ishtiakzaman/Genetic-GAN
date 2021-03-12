# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import imageio
import numpy as np
import random

trial = 5
epoch = 101
offspring_mode = "mean"

f_file = open('test_good_female.txt', 'r')
m_file = open('test_good_male.txt', 'r')

f_list = []
m_list = []

for line in f_file:
    f_list.append(line.strip())
for line in m_file:
    m_list.append(line.strip())
print(len(f_list), len(m_list))

random.shuffle(f_list)
random.shuffle(m_list)

parent_image_root = '../img_align_celeba/high/'
child_image_root = 'test_good_dump/' + offspring_mode + '/'
mturk_root = 'mturk_photos_1/'

for ix in range(min(len(f_list), len(m_list))):
    #f_img = f_list[int(np.random.rand() * len(f_list))] + '.jpg'
    #m_img = m_list[int(np.random.rand() * len(m_list))] + '.jpg'
    f_img = f_list[ix] + '.jpg'
    m_img = m_list[ix] + '.jpg'
    cg = ['f', 'm'][int(round(np.random.rand()))]
    ci = int(np.random.rand() * 5)
    c_img = f_img[:-4] + '_' + m_img[:-4] + '_' + cg + '_' + str(ci) + '.jpg' 
    if os.path.isfile(child_image_root + c_img) == False:
        continue

    cmd = 'montage -label Mother ' + parent_image_root + f_img + ' null: -label Father ' + parent_image_root + m_img
    cmd = cmd + ' null: -label Child ' + child_image_root + c_img + ' null: -tile 3x2 -geometry 89x129+10+10 '
    cmd = cmd + mturk_root + f_img[:-4] + '_' + m_img[:-4] + '_' + cg + '_' + str(ci) + '.jpg'
    #print(cmd)
    os.system(cmd)

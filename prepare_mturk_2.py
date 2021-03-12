# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import imageio
import numpy as np

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

parent_image_root = '../img_align_celeba/high/'
child_image_root = 'test_good_dump/' + offspring_mode + '/'
mturk_root = 'mturk_photos_2/'

for serial in range(1000):
    f_img_original = f_list[int(np.random.rand() * len(f_list))] + '.jpg'
    m_img_original = m_list[int(np.random.rand() * len(m_list))] + '.jpg'
    cg = ['f', 'm'][int(round(np.random.rand()))]
    ci = int(np.random.rand() * 5)
    c_img_original = f_img_original[:-4] + '_' + m_img_original[:-4] + '_' + cg + '_' + str(ci) + '.jpg' 
    if os.path.isfile(child_image_root + c_img_original) == False:
        continue

    # Make 4 other children from 4 different set of parents.
    c_img_list = []
    while True:
        f_img_other = f_list[int(np.random.rand() * len(f_list))] + '.jpg'
        m_img_other = m_list[int(np.random.rand() * len(m_list))] + '.jpg'
        if f_img_other != f_img_original and m_img_other != m_img_original:
            cg = ['f', 'm'][int(round(np.random.rand()))]
            ci = int(np.random.rand() * 5)
            c_img_other = f_img_other[:-4] + '_' + m_img_other[:-4] + '_' + cg + '_' + str(ci) + '.jpg' 
            if os.path.isfile(child_image_root + c_img_other) == False:
                continue
            c_img_list.append(c_img_other)
            if len(c_img_list) == 4:
                break

    # Making random original child index.
    ci = int(np.random.rand() * 5)
    c_img_list.insert(ci, c_img_original)
        
    cmd = 'montage null: -label Mother ' + parent_image_root + f_img_original + ' null: -label Father ' + parent_image_root + m_img_original + ' null:'
    for ix, c_img in enumerate(c_img_list, 1):
        cmd = cmd + ' -label ' + str(ix) + ' '  + child_image_root + c_img
    cmd = cmd + ' -tile 5x2 -geometry 89x129+10+10 '
    cmd = cmd + mturk_root + str(serial) + '_' + str(ci) + '.jpg'
    #print(cmd)
    os.system(cmd)
    #break

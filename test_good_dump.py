# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
import model_high as model
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover
from random import shuffle
from scipy import ndimage

# Good Results:
# 5_101. mean
# 8_126. split
# 9_91. part

trial = 5
epoch = 101

device = torch.device("cuda:0")

#offspring_mode = "part"
#offspring_mode = "split"
offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

b_size = 32

f_data = DatasetHDF5("dataset/female_young_old_test.h5", b_size, True, 100, ['image', 'name'])
m_data = DatasetHDF5("dataset/male_young_old_test.h5", b_size, True, 100, ['image', 'name'])

Encoder = model.Encoder().to(device)
Encoder.load_state_dict(torch.load('weight_high/' + str(trial) + '_encoder_' + str(epoch) + '.pt', map_location='cpu'))
Encoder.eval()

DecoderFemale = model.Decoder().to(device)
DecoderFemale.load_state_dict(torch.load('weight_high/' + str(trial) + '_decoder_f_' + str(epoch) + '.pt', map_location='cpu'))
DecoderFemale.eval()

DecoderMale = model.Decoder().to(device)
DecoderMale.load_state_dict(torch.load('weight_high/' + str(trial) + '_decoder_m_' +  str(epoch) + '.pt', map_location='cpu'))
DecoderMale.eval()

for child in Encoder.children():
    #for ii in range(len(child)):
    if type(child)==nn.BatchNorm2d:
        child.track_running_stats = False
for child in DecoderFemale.children():
    #for ii in range(len(child)):
    if type(child)==nn.BatchNorm2d:
        child.track_running_stats = False
for child in DecoderMale.children():
    #for ii in range(len(child)):
    if type(child)==nn.BatchNorm2d:
        child.track_running_stats = False

serial = 1

'''
# First pass to stabilize running mean etc. probably no use at all.
while True:
    epoch_happened, f_in_img, f_in_labels = f_data.load_batch('train')
    _, m_in_img, m_in_labels = m_data.load_batch('train')

    if epoch_happened == True:
        break

    if f_in_img.shape[0] != b_size or m_in_img.shape[0] != b_size:
        continue

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    for ci in xrange(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring)
        offspring_m_out_img = DecoderMale(z_offspring)
'''

f_file = open('test_good_female.txt', 'r')
m_file = open('test_good_male.txt', 'r')

f_list = []
m_list = []

for line in f_file:
    f_list.append(line.strip())
for line in m_file:
    m_list.append(line.strip())
print(len(f_list), len(m_list))

f_list = ['004180']
m_list = ['084071']

index_list = []
for i in xrange(len(f_list)):
    for j in xrange(len(m_list)):
        index_list.append((i, j))
shuffle(index_list)
print(len(index_list))


image_root = '../img_align_celeba/high/'

serial = 1
index = 0

#while index + b_size < len(index_list):
b_size = 1
while True:

    f_name = [f_list[index_list[index+i][0]] for i in xrange(b_size)]
    m_name = [m_list[index_list[index+i][1]] for i in xrange(b_size)]
    index = index + b_size

    f_in_img = [ndimage.imread(image_root + f_name[i] + '.jpg', mode='RGB') for i in xrange(b_size)]
    m_in_img = [ndimage.imread(image_root + m_name[i] + '.jpg', mode='RGB') for i in xrange(b_size)]
    f_in_img = np.stack(f_in_img, axis=0)
    m_in_img = np.stack(m_in_img, axis=0)

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    for ci in xrange(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring)
        offspring_m_out_img = DecoderMale(z_offspring)
        offspring_f_out_img = postprocess_convert_images(offspring_f_out_img.detach().cpu())
        offspring_m_out_img = postprocess_convert_images(offspring_m_out_img.detach().cpu())
        
        for i in xrange(b_size):
            imsave('test_good_dump/' + offspring_mode + '/' + f_name[i] + '_' + m_name[i] + '_f_' + str(ci) + '.jpg', offspring_f_out_img[i])
            imsave('test_good_dump/' + offspring_mode + '/' + f_name[i] + '_' + m_name[i] + '_m_' + str(ci) + '.jpg', offspring_m_out_img[i])

    break

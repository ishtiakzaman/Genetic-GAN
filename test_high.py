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

# Good Results:
# 5_101. mean
# 8_126. split
# 9_91. part

trial = 5
epoch = 101

device = torch.device("cuda:3")

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

DecoderFemale = model.Decoder().to(device)
DecoderFemale.load_state_dict(torch.load('weight_high/' + str(trial) + '_decoder_f_' + str(epoch) + '.pt', map_location='cpu'))

DecoderMale = model.Decoder().to(device)
DecoderMale.load_state_dict(torch.load('weight_high/' + str(trial) + '_decoder_m_' +  str(epoch) + '.pt', map_location='cpu'))

'''
Encoder.eval()
DecoderFemale.eval()
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

DiscriminatorFemale = model.Discriminator().to(device)
DiscriminatorFemale.load_state_dict(torch.load('weight_high/' + str(trial) + '_discriminator_f_' + str(epoch) + '.pt', map_location='cpu'))

DiscriminatorMale = model.Discriminator().to(device)
DiscriminatorMale.load_state_dict(torch.load('weight_high/' + str(trial) + '_discriminator_f_' + str(epoch) + '.pt', map_location='cpu'))

'''
#output_list = []
serial = 1

'''
# First pass to stabilize running mean etc.
while True:
    epoch_happened, f_in_img, f_in_labels = f_data.load_batch('train')
    _, m_in_img, m_in_labels = m_data.load_batch('train')

    if epoch_happened == True:
        break

    if f_in_img.shape[0] != b_size or m_in_img.shape[0] != b_size:
        continue

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    with torch.no_grad():
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
    for ci in xrange(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        with torch.no_grad():
            offspring_f_out_img = DecoderFemale(z_offspring)
            offspring_m_out_img = DecoderMale(z_offspring)
'''

# Second pass, actaul work.
while True:
    epoch_happened, f_in_img, f_in_labels = f_data.load_batch('train')
    _, m_in_img, m_in_labels = m_data.load_batch('train')

    if epoch_happened == True:
        break

    if f_in_img.shape[0] != b_size or m_in_img.shape[0] != b_size:
        continue


    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    #disc_value = 0
    output = []
    Encoder.zero_grad()
    DecoderFemale.zero_grad()
    DecoderMale.zero_grad()
    with torch.no_grad():
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        output.append(postprocess_convert_images(f_in_img.detach().cpu()))
        output.append(postprocess_convert_images(m_in_img.detach().cpu()))
    for ci in xrange(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        with torch.no_grad():
            offspring_f_out_img = DecoderFemale(z_offspring)
            offspring_m_out_img = DecoderMale(z_offspring)
            #disc_value = disc_value + DiscriminatorFemale(offspring_f_out_img).detach().cpu().numpy().mean()
            #disc_value = disc_value + DiscriminatorMale(offspring_m_out_img).detach().cpu().numpy().mean()
            output.append(postprocess_convert_images(offspring_f_out_img.detach().cpu()))
            output.append(postprocess_convert_images(offspring_m_out_img.detach().cpu()))
    #output.append(disc_value)
    #output_list.append(output)
    #if len(output_list) > 50:
    #    break

    #output_list.sort(key=lambda tup: tup[12], reverse=True)
    #shuffle(output_list)

    n_row = b_size
    n_col = 12

    img_row = 218
    img_col = 178
    draw_string = ''    
    image = np.zeros((img_row*n_row, img_col*n_col + 350, 3), dtype=np.uint8)
    for i in xrange(n_row):
        image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output[0][i]
        image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output[1][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output[2+ci*2][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output[3+ci*2][i]

        draw_string = draw_string + 'text 2200,' + str(100 + i * img_row) + ' \'' + f_in_labels[i][0] + '\' '
        draw_string = draw_string + 'text 2200,' + str(150 + i * img_row) + ' \'' + m_in_labels[i][0] + '\' '
        
    imname = 'demo/sample_' + str(trial) + '_' + str(epoch) + '_' + str(serial) + '.jpg'
    imsave(imname, image)
    os.system('convert ' + imname + ' -stroke white -fill white -pointsize 36 -font Rachana-Regular -draw \"' + draw_string + '\" ' + imname)
    serial = serial + 1
    break

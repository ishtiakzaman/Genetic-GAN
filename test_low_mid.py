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
import model
import model_mid
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover
from random import shuffle

trial = 6
epoch = 98

device = torch.device("cuda:5")

#offspring_mode = "part"
#offspring_mode = "split"
offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 10

f_data = DatasetHDF5("dataset/female_test_mid.h5", b_size, True, 100, ['image'])
m_data = DatasetHDF5("dataset/male_test_mid.h5", b_size, True, 100, ['image'])

Encoder = model.Encoder().to(device)
Encoder.load_state_dict(torch.load('weight/encoder_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DecoderFemale = model.Decoder().to(device)
DecoderFemale.load_state_dict(torch.load('weight/decoder_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DecoderMale = model.Decoder().to(device)
DecoderMale.load_state_dict(torch.load('weight/decoder_m_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

GeneratorFemaleMid = model_mid.Generator().to(device)
GeneratorFemaleMid.load_state_dict(torch.load('weight/generator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

GeneratorMaleMid = model_mid.Generator().to(device)
GeneratorMaleMid.load_state_dict(torch.load('weight/generator_mid_m_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DiscriminatorFemaleMid = model_mid.Discriminator().to(device)
DiscriminatorFemaleMid.load_state_dict(torch.load('weight/discriminator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DiscriminatorMaleMid = model_mid.Discriminator().to(device)
DiscriminatorMaleMid.load_state_dict(torch.load('weight/discriminator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

#output_list = []
serial = 1

while True:
    epoch_happened, f_in_img_mid = f_data.load_batch('train')
    _, m_in_img_mid = m_data.load_batch('train')

    if epoch_happened == True:
        break

    if f_in_img_mid.shape[0] != b_size or m_in_img_mid.shape[0] != b_size:
        continue

    f_in_img = np.empty((b_size, 55, 45, 3), np.float)
    m_in_img = np.empty((b_size, 55, 45, 3), np.float)

    for i in xrange(b_size):
        f_in_img[i] = resize(f_in_img_mid[i], (55, 45, 3), preserve_range=True)
        m_in_img[i] = resize(m_in_img_mid[i], (55, 45, 3), preserve_range=True)
        if np.random.random() > 0.5:
            f_in_img[i] = np.fliplr(f_in_img[i])
            f_in_img_mid[i] = np.fliplr(f_in_img_mid[i])
        if np.random.random() > 0.5:
            m_in_img[i] = np.fliplr(m_in_img[i])
            m_in_img_mid[i] = np.fliplr(m_in_img_mid[i])

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    #disc_value = 0
    output = []
    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    output.append(f_in_img_mid)
    output.append(m_in_img_mid)
    for ci in xrange(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring)
        offspring_m_out_img = DecoderMale(z_offspring)
        offspring_f_out_img_mid = GeneratorFemaleMid(offspring_f_out_img)
        offspring_m_out_img_mid = GeneratorMaleMid(offspring_m_out_img)
        #disc_value = disc_value + DiscriminatorFemaleMid(offspring_f_out_img_mid).detach().cpu().numpy().mean()
        #disc_value = disc_value + DiscriminatorMaleMid(offspring_m_out_img_mid).detach().cpu().numpy().mean()
        output.append(postprocess_convert_images(offspring_f_out_img_mid.detach().cpu()))
        output.append(postprocess_convert_images(offspring_m_out_img_mid.detach().cpu()))
    #output.append(disc_value)
    #output_list.append(output)
    #if len(output_list) > 50:
    #    break

    #output_list.sort(key=lambda tup: tup[12], reverse=True)
    #shuffle(output_list)

    n_row = b_size
    n_col = 12
    img_row = 109
    img_col = 89
    image = np.zeros((img_row*n_row, img_col*n_col, 3), dtype=np.uint8)
    for i in xrange(n_row):
        image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output[0][i]
        image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output[1][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output[2+ci*2][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output[3+ci*2][i]
    imsave('sample_test/low_mid_' + str(trial) + '_' + str(epoch) + '_' + str(serial) + '.jpg', image)
    serial = serial + 1

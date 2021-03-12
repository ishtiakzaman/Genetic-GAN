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
import model_high
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover
from random import shuffle

trial = 3
epoch = 165

device = torch.device("cuda:7")

#offspring_mode = "part"
#offspring_mode = "split"
offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 10

f_data = DatasetHDF5("dataset/female_high_1.h5", b_size, True, 100, ['image'])
m_data = DatasetHDF5("dataset/male_high_1.h5", b_size, True, 100, ['image'])

Encoder = model.Encoder().to(device)
Encoder.load_state_dict(torch.load('weight_low_mid_high/encoder_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DecoderFemale = model.Decoder().to(device)
DecoderFemale.load_state_dict(torch.load('weight_low_mid_high/decoder_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DecoderMale = model.Decoder().to(device)
DecoderMale.load_state_dict(torch.load('weight_low_mid_high/decoder_m_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

GeneratorFemaleMid = model_mid.Generator().to(device)
GeneratorFemaleMid.load_state_dict(torch.load('weight_low_mid_high/generator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

GeneratorMaleMid = model_mid.Generator().to(device)
GeneratorMaleMid.load_state_dict(torch.load('weight_low_mid_high/generator_mid_m_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

GeneratorFemaleHigh = model_high.Generator().to(device)
GeneratorFemaleHigh.load_state_dict(torch.load('weight_low_mid_high/generator_high_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

GeneratorMaleHigh = model_high.Generator().to(device)
GeneratorMaleHigh.load_state_dict(torch.load('weight_low_mid_high/generator_high_m_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DiscriminatorFemaleHigh = model_high.Discriminator().to(device)
DiscriminatorFemaleHigh.load_state_dict(torch.load('weight_low_mid_high/discriminator_high_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

DiscriminatorMaleHigh = model_high.Discriminator().to(device)
DiscriminatorMaleHigh.load_state_dict(torch.load('weight_low_mid_high/discriminator_high_f_' + str(trial) + '_' + str(epoch) + '.pt', map_location='cpu'))

#output_list = []
serial = 1

while True:
    epoch_happened, f_in_img_high = f_data.load_batch('train')
    _, m_in_img_high = m_data.load_batch('train')

    if epoch_happened == True:
        break

    if f_in_img_high.shape[0] != b_size or m_in_img_high.shape[0] != b_size:
        continue

    f_in_img_mid = np.empty((b_size, 109, 89, 3), np.float)
    m_in_img_mid = np.empty((b_size, 109, 89, 3), np.float)

    f_in_img = np.empty((b_size, 55, 45, 3), np.float)
    m_in_img = np.empty((b_size, 55, 45, 3), np.float)

    for i in xrange(b_size):
        f_in_img_mid[i] = resize(f_in_img_high[i], (109, 89, 3), preserve_range=True)
        m_in_img_mid[i] = resize(m_in_img_high[i], (109, 89, 3), preserve_range=True)
        f_in_img[i] = resize(f_in_img_high[i], (55, 45, 3), preserve_range=True)
        m_in_img[i] = resize(m_in_img_high[i], (55, 45, 3), preserve_range=True)
        if np.random.random() > 0.5:
            f_in_img[i] = np.fliplr(f_in_img[i])
            f_in_img_mid[i] = np.fliplr(f_in_img_mid[i])
            f_in_img_high[i] = np.fliplr(f_in_img_high[i])
        if np.random.random() > 0.5:
            m_in_img[i] = np.fliplr(m_in_img[i])
            m_in_img_mid[i] = np.fliplr(m_in_img_mid[i])
            m_in_img_high[i] = np.fliplr(m_in_img_high[i])

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    #disc_value = 0
    output_mid = []
    output_high = []
    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    output_mid.append(f_in_img_mid)
    output_mid.append(m_in_img_mid)
    output_high.append(f_in_img_high)
    output_high.append(m_in_img_high)
    for ci in xrange(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring)
        offspring_m_out_img = DecoderMale(z_offspring)
        offspring_f_out_img_mid = GeneratorFemaleMid(offspring_f_out_img)
        offspring_m_out_img_mid = GeneratorMaleMid(offspring_m_out_img)
        offspring_f_out_img_high = GeneratorFemaleHigh(offspring_f_out_img_mid)
        offspring_m_out_img_high = GeneratorMaleHigh(offspring_m_out_img_mid)
        #disc_value = disc_value + DiscriminatorFemaleMid(offspring_f_out_img_mid).detach().cpu().numpy().mean()
        #disc_value = disc_value + DiscriminatorMaleMid(offspring_m_out_img_mid).detach().cpu().numpy().mean()
        output_mid.append(postprocess_convert_images(offspring_f_out_img_mid.detach().cpu()))
        output_mid.append(postprocess_convert_images(offspring_m_out_img_mid.detach().cpu()))
        output_high.append(postprocess_convert_images(offspring_f_out_img_high.detach().cpu()))
        output_high.append(postprocess_convert_images(offspring_m_out_img_high.detach().cpu()))
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
        image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output_mid[0][i]
        image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output_mid[1][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output_mid[2+ci*2][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output_mid[3+ci*2][i]
    imsave('sample_test/low_mid_high_MID_' + str(trial) + '_' + str(epoch) + '_' + str(serial) + '.jpg', image)

    img_row = 218
    img_col = 178
    image = np.zeros((img_row*n_row, img_col*n_col, 3), dtype=np.uint8)
    for i in xrange(n_row):
        image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output_high[0][i]
        image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output_high[1][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output_high[2+ci*2][i]
        for ci in xrange(5):
            image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output_high[3+ci*2][i]
    imsave('sample_test/low_mid_high_HIGH_' + str(trial) + '_' + str(epoch) + '_' + str(serial) + '.jpg', image)
    serial = serial + 1

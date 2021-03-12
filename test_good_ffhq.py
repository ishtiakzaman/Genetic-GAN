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
import model_high as model
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover
from random import shuffle
import imageio

trial = 5
epoch = 101

device = torch.device("cuda:1")

#offspring_mode = "part"
#offspring_mode = "split"
offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

b_size = 16

Encoder = model.Encoder().to(device)
Encoder.load_state_dict(torch.load('weight_high/' + str(trial) + '_encoder_' + str(epoch) + '.pt', map_location='cpu'))

DecoderFemale = model.Decoder().to(device)
DecoderFemale.load_state_dict(torch.load('weight_high/' + str(trial) + '_decoder_f_' + str(epoch) + '.pt', map_location='cpu'))

DecoderMale = model.Decoder().to(device)
DecoderMale.load_state_dict(torch.load('weight_high/' + str(trial) + '_decoder_m_' +  str(epoch) + '.pt', map_location='cpu'))

sg_file = open('../stylegan_truncation_tricks_list.txt', 'r')
sg_list = []

for line in sg_file:
    sg_list.append(line.strip())

image_root = '../img_align_celeba/high/'
sg_root = '../stylegan_truncation_tricks_178x178/000000-psi-1/'

serial = 1

def read_image(root, file_name):
    img = imageio.imread(root + file_name + '.jpg')
    img_rev = np.flipud(img)
    image = np.concatenate((img_rev, img), axis=0)
    image = np.concatenate((image, img_rev), axis=0)
    image = image[150:150+218, :, :]
    return image

# demo looping for batch_normalization avg.
for _ in range(2000):

    f_name = [sg_list[int(np.random.random() * len(sg_list))][:-4] for _ in range(b_size)]
    m_name = [sg_list[int(np.random.random() * len(sg_list))][:-4]for _ in range(b_size)]

    f_in_img = [read_image(sg_root, f_name[i]) for i in range(b_size)]
    m_in_img = [read_image(sg_root, m_name[i]) for i in range(b_size)]
    f_in_img = np.stack(f_in_img, axis=0)
    m_in_img = np.stack(m_in_img, axis=0)

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    output = []
    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    output.append(postprocess_convert_images(f_in_img.detach().cpu()))
    output.append(postprocess_convert_images(m_in_img.detach().cpu()))
    for ci in range(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring)
        offspring_m_out_img = DecoderMale(z_offspring)
        output.append(postprocess_convert_images(offspring_f_out_img.detach().cpu()))
        output.append(postprocess_convert_images(offspring_m_out_img.detach().cpu()))

    n_row = b_size
    n_col = 12

    '''
    img_row = 218
    img_col = 178
    draw_string = ''    
    image = np.zeros((img_row*n_row, img_col*n_col + 350, 3), dtype=np.uint8)
    for i in range(n_row):
        image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output[0][i]
        image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output[1][i]
        for ci in range(5):
            image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output[2+ci*2][i]
        for ci in range(5):
            image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output[3+ci*2][i]

        draw_string = draw_string + 'text 2200,' + str(100 + i * img_row) + ' \'' + f_name[i] + '\' '
        draw_string = draw_string + 'text 2200,' + str(150 + i * img_row) + ' \'' + m_name[i] + '\' '
        
    imname = 'test_ffhq/sample_' + str(trial) + '_' + str(epoch) + '_' + str(serial) + '.jpg'
    imageio.imwrite(imname, image)
    os.system('convert ' + imname + ' -stroke white -fill white -pointsize 36 -font Rachana-Regular -draw \"' + draw_string + '\" ' + imname)
    serial = serial + 1
    '''

good_female = ['000342', '000547', '000977', '000730', '000003', '000181', '000627', '000652', '000642', '000593', '000398', '000977', '000087', '000306', '000009', '000537', '000878']
good_male = ['000832', '000949', '000424', '000375', '000173', '000207', '000700', '000063', '000161', '000558', '000479', '000918', '000048', '000927', '000713']

for _ in range(500):

    f_name = [good_female[int(np.random.random() * len(good_female))] for _ in range(b_size)]
    m_name = [good_male[int(np.random.random() * len(good_male))]for _ in range(b_size)]

    f_in_img = [read_image(sg_root, f_name[i]) for i in range(b_size)]
    m_in_img = [read_image(sg_root, m_name[i]) for i in range(b_size)]
    f_in_img = np.stack(f_in_img, axis=0)
    m_in_img = np.stack(m_in_img, axis=0)

    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    output = []
    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    output.append(postprocess_convert_images(f_in_img.detach().cpu()))
    output.append(postprocess_convert_images(m_in_img.detach().cpu()))
    for ci in range(5):
        z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring)
        offspring_m_out_img = DecoderMale(z_offspring)
        output.append(postprocess_convert_images(offspring_f_out_img.detach().cpu()))
        output.append(postprocess_convert_images(offspring_m_out_img.detach().cpu()))

    n_row = b_size
    n_col = 12

    img_row = 218
    img_col = 178
    draw_string = ''    
    image = np.zeros((img_row*n_row, img_col*n_col + 350, 3), dtype=np.uint8)
    for i in range(n_row):
        image[i*img_row:(i+1)*img_row, img_col*0:img_col*1, :] = output[0][i]
        image[i*img_row:(i+1)*img_row, img_col*1:img_col*2, :] = output[1][i]
        for ci in range(5):
            image[i*img_row:(i+1)*img_row, img_col*2+img_col*ci:img_col*2+img_col*(ci+1), :] = output[2+ci*2][i]
        for ci in range(5):
            image[i*img_row:(i+1)*img_row, img_col*7+img_col*ci:img_col*7+img_col*(ci+1), :] = output[3+ci*2][i]

        draw_string = draw_string + 'text 2200,' + str(100 + i * img_row) + ' \'' + f_name[i] + '\' '
        draw_string = draw_string + 'text 2200,' + str(150 + i * img_row) + ' \'' + m_name[i] + '\' '
        
    imname = 'test_ffhq/sample_good_' + str(trial) + '_' + str(epoch) + '_' + str(serial) + '.jpg'
    imageio.imwrite(imname, image)
    os.system('convert ' + imname + ' -stroke white -fill white -pointsize 36 -font Rachana-Regular -draw \"' + draw_string + '\" ' + imname)
    serial = serial + 1

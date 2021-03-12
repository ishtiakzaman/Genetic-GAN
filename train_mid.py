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
from skimage import img_as_ubyte
import model_mid as model
import model as model_low
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover


# NOT GOOD RESULT

trial = 2

device = torch.device("cuda:6")

#offspring_mode = "part"
#offspring_mode = "split"
offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 32
patch_len = (1, 4 ,3)
num_epochs = 300

lr = 0.0002
beta1 = 0.5

f_data = DatasetHDF5("dataset/female_mid_1.h5", b_size, True, 100, ['image'])
m_data = DatasetHDF5("dataset/male_mid_1.h5", b_size, True, 100, ['image'])
disc_f_data = DatasetHDF5("dataset/female_mid_2.h5", b_size, True, 100, ['image'])
disc_m_data = DatasetHDF5("dataset/male_mid_2.h5", b_size, True, 100, ['image'])

# custom weights initialization called on Encoder and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

Encoder = model_low.Encoder().to(device)
Encoder.load_state_dict(torch.load('weight/encoder_79_95.pt', map_location='cpu'))
DecoderFemale = model_low.Decoder().to(device)
DecoderFemale.load_state_dict(torch.load('weight/decoder_f_79_95.pt', map_location='cpu'))
DecoderMale = model_low.Decoder().to(device)
DecoderMale.load_state_dict(torch.load('weight/decoder_m_79_95.pt', map_location='cpu'))

GeneratorFemale = model.Generator().to(device)
GeneratorFemale.apply(weights_init)

GeneratorMale = model.Generator().to(device)
GeneratorMale.apply(weights_init)

DiscriminatorFemale = model.Discriminator().to(device)
DiscriminatorFemale.apply(weights_init)

DiscriminatorMale = model.Discriminator().to(device)
DiscriminatorMale.apply(weights_init)

loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()

optimizerGeneratorFemale = optim.Adam(GeneratorFemale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerGeneratorMale = optim.Adam(GeneratorMale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemale = optim.Adam(DiscriminatorFemale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorMale = optim.Adam(DiscriminatorMale.parameters(), lr=lr, betas=(beta1, 0.999))

iters = 0

print("Starting Training Loop...")
    
log_file = open('log/log_mid_' + str(trial) + '.txt', 'w')
graph_file = open('graph/graph_mid_' + str(trial) + '.csv', 'w')
graph_file.write('iters,real,fake,G\n')
epoch = 1
child_index = 0


while True:
    epoch_happened, f_in_img = f_data.load_batch('train')
    _, m_in_img = m_data.load_batch('train')
    _, disc_f_in_img = disc_f_data.load_batch('train')
    _, disc_m_in_img = disc_m_data.load_batch('train')

    if f_in_img.shape[0] != b_size or m_in_img.shape[0] != b_size:
        continue
    if disc_f_in_img.shape[0] != b_size or disc_m_in_img.shape[0] != b_size:
        continue

    f_in_img_low = np.empty((b_size, 55, 45, 3), np.float)
    m_in_img_low = np.empty((b_size, 55, 45, 3), np.float)
    #disc_f_in_img_low = np.empty((b_size, 55, 45, 3), np.float)
    #disc_m_in_img_low = np.empty((b_size, 55, 45, 3), np.float)

    for i in xrange(b_size):
        f_in_img_low[i] = resize(f_in_img[i], (55, 45, 3), preserve_range=True)
        m_in_img_low[i] = resize(m_in_img[i], (55, 45, 3), preserve_range=True)
        #disc_f_in_img_low[i] = resize(disc_f_in_img[i], (55, 45, 3), preserve_range=True)
        #disc_m_in_img_low[i] = resize(disc_m_in_img[i], (55, 45, 3), preserve_range=True)
        if np.random.random() > 0.5:
            f_in_img[i] = np.fliplr(f_in_img[i])
            f_in_img_low[i] = np.fliplr(f_in_img_low[i])
        if np.random.random() > 0.5:
            m_in_img[i] = np.fliplr(m_in_img[i])
            m_in_img_low[i] = np.fliplr(m_in_img_low[i])
        if np.random.random() > 0.5:
            disc_f_in_img[i] = np.fliplr(disc_f_in_img[i])
            #disc_f_in_img_low[i] = np.fliplr(disc_f_in_img_low[i])
        if np.random.random() > 0.5:
            disc_m_in_img[i] = np.fliplr(disc_m_in_img[i])
            #disc_m_in_img_low[i] = np.fliplr(disc_m_in_img_low[i])

    disc_f_in_img = preprocess_convert_images(disc_f_in_img).to(device)
    disc_m_in_img = preprocess_convert_images(disc_m_in_img).to(device)
    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    #disc_f_in_img_low = preprocess_convert_images(disc_f_in_img_low).to(device)
    #disc_m_in_img_low = preprocess_convert_images(disc_m_in_img_low).to(device)
    f_in_img_low = preprocess_convert_images(f_in_img_low).to(device)
    m_in_img_low = preprocess_convert_images(m_in_img_low).to(device)

    zero_label = torch.full((b_size, patch_len[0], patch_len[1], patch_len[2]), 0, device=device)    
    one_label = torch.full((b_size, patch_len[0], patch_len[1], patch_len[2]), 1, device=device)    

    '''
    with torch.no_grad():
        z_female = Encoder(disc_f_in_img_low)
        z_male = Encoder(disc_m_in_img_low)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        disc_offspring_f_low = DecoderFemale(z_offspring).detach()
        disc_offspring_m_low = DecoderMale(z_offspring).detach()
    '''

    with torch.no_grad():
        z_female = Encoder(f_in_img_low)
        z_male = Encoder(m_in_img_low)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        offspring_f_low = DecoderFemale(z_offspring).detach()
        offspring_m_low = DecoderMale(z_offspring).detach()

    # Female Discriminator loss.
    DiscriminatorFemale.zero_grad()
    disc_f_out = DiscriminatorFemale(disc_f_in_img)
    loss_real_D_female = loss_mse(disc_f_out, one_label)
    loss_real_D_female.backward()

    disc_m_out = DiscriminatorFemale(disc_m_in_img)

    with torch.no_grad():
        offspring_f_mid = GeneratorFemale(offspring_f_low).detach()
    disc_offspring_f_out = DiscriminatorFemale(offspring_f_mid)

    loss_fake_D_female = loss_mse(torch.cat((disc_m_out, disc_offspring_f_out), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_female.backward()

    optimizerDiscriminatorFemale.step()

    # Male Discriminator loss.
    DiscriminatorMale.zero_grad()
    disc_m_out = DiscriminatorMale(disc_m_in_img)
    loss_real_D_male = loss_mse(disc_m_out, one_label)
    loss_real_D_male.backward()

    disc_f_out = DiscriminatorMale(disc_f_in_img)

    with torch.no_grad():
        offspring_m_mid = GeneratorMale(offspring_m_low).detach()
    disc_offspring_m_out = DiscriminatorMale(offspring_m_mid)

    loss_fake_D_male = loss_mse(torch.cat((disc_f_out, disc_offspring_m_out), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_male.backward()

    optimizerDiscriminatorMale.step()

    # Female Offspring Adversarial loss.
    GeneratorFemale.zero_grad()
    offspring_f_mid = GeneratorFemale(offspring_f_low)
    disc_offspring_f_out = DiscriminatorFemale(offspring_f_mid)

    loss_G_female = loss_mse(disc_offspring_f_out, one_label)
    loss_G_female.backward()

    optimizerGeneratorFemale.step()

    # Male Offspring Adversarial loss.
    GeneratorMale.zero_grad()
    offspring_m_mid = GeneratorMale(offspring_m_low)
    disc_offspring_m_out = DiscriminatorMale(offspring_m_mid)

    loss_G_male = loss_mse(disc_offspring_m_out, one_label)
    loss_G_male.backward()

    optimizerGeneratorMale.step()

    # Output training stats
    if iters % 50 == 0:
        print_str = '[trial: %d][iter: %d][epoch: %d/%d]\tL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f' % (trial, iters, epoch, num_epochs, loss_real_D_female.item(), loss_real_D_male.item(), loss_fake_D_female.item(), loss_fake_D_male.item(), loss_G_female.item(), loss_G_male.item())
        print(print_str)
        log_file.write(print_str + '\n')
        log_file.flush()
        graph_str = str(iters) + ',' + str((loss_real_D_female.item() + loss_real_D_male.item()) * 0.5) + ',' + str((loss_fake_D_female.item() + loss_fake_D_male.item()) * 0.5) + ',' + str((loss_G_female.item() + loss_G_male.item()) * 0.5)
        graph_file.write(graph_str + '\n')
        graph_file.flush()
    
    iters += 1
    child_index = (child_index + 1) % 5
    if epoch_happened == True:
    #if iters > 1 and iters % 50 == 1:
        n_row = 10
        n_col_low = 10
        n_col_mid = 12
        f_in_img = postprocess_convert_images(f_in_img.cpu())
        m_in_img = postprocess_convert_images(m_in_img.cpu())
        f_in_img_low = postprocess_convert_images(f_in_img_low.cpu())
        m_in_img_low = postprocess_convert_images(m_in_img_low.cpu())
        offspring_female_list_low = []
        offspring_male_list_low = []
        offspring_female_list_mid = []
        offspring_male_list_mid = []
        for ci in xrange(5):
            #z_female = Encoder(f_in_img_low)
            #z_male = Encoder(m_in_img_low)
            offspring_female_list_low.append(postprocess_convert_images(DecoderFemale(crossover(z_female, z_male, ci, offspring_mode, device)).detach().cpu()))
            offspring_male_list_low.append(postprocess_convert_images(DecoderMale(crossover(z_female, z_male, ci, offspring_mode, device)).detach().cpu()))
            offspring_female_list_mid.append(postprocess_convert_images(GeneratorFemale(DecoderFemale(crossover(z_female, z_male, ci, offspring_mode, device))).detach().cpu()))
            offspring_male_list_mid.append(postprocess_convert_images(GeneratorMale(DecoderMale(crossover(z_female, z_male, ci, offspring_mode, device))).detach().cpu()))
        img_row_low = 55
        img_col_low = 45
        img_row_mid = 109
        img_col_mid = 89
        image = np.zeros((img_row_mid*n_row, img_col_low*n_col_low+img_col_mid*n_col_mid, 3), dtype=np.uint8)
        for r in xrange(n_row):
            image[r*img_row_mid:(r+1)*img_row_mid, img_col_mid*0:img_col_mid*1, :] = f_in_img[r]
            image[r*img_row_mid:(r+1)*img_row_mid, img_col_mid*1:img_col_mid*2, :] = m_in_img[r]
            #image[r*img_row_mid:r*img_row_mid+img_row_low, img_col_mid*2:img_col_mid*2+img_col_low, :] = f_in_img_low[r]
            #image[r*img_row_mid:r*img_row_mid+img_row_low, img_col_mid*2+img_col_low:img_col_mid*2+2*img_col_low, :] = m_in_img_low[r]
            for ci in xrange(5):
                image[r*img_row_mid:r*img_row_mid+img_row_low, img_col_mid*2+img_col_low*ci:img_col_mid*2+img_col_low*(ci+1), :] = offspring_female_list_low[ci][r]
            for ci in xrange(5):
                image[r*img_row_mid:r*img_row_mid+img_row_low, img_col_mid*2+img_col_low*(ci+5):img_col_mid*2+img_col_low*(ci+6), :] = offspring_male_list_low[ci][r]
            for ci in xrange(5):
                image[r*img_row_mid:(r+1)*img_row_mid, img_col_mid*2+img_col_low*10+img_col_mid*ci:img_col_mid*2+img_col_low*10+img_col_mid*(ci+1), :] = offspring_female_list_mid[ci][r]
            for ci in xrange(5):
                image[r*img_row_mid:(r+1)*img_row_mid, img_col_mid*2+img_col_low*10+img_col_mid*(ci+5):img_col_mid*2+img_col_low*10+img_col_mid*(ci+6), :] = offspring_male_list_mid[ci][r]
        imsave('sample_mid/train_' + str(trial) + "_" + str(epoch) + '.jpg', image)    
        #if epoch % 5 == 0:
        torch.save(GeneratorFemale.state_dict(), 'weight/generator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(GeneratorMale.state_dict(), 'weight/generator_mid_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorFemale.state_dict(), 'weight/discriminator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorMale.state_dict(), 'weight/discriminator_mid_m_' + str(trial) + '_' + str(epoch) + '.pt')
        epoch += 1
        if epoch > num_epochs:
            break

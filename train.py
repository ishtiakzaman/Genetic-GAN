# -*- coding: utf-8 -*-
from __future__ import print_function
#%matplotlib inline
import argparse
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
import model
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images

# Trial 1: hparam_kl = 1.0, hparam_rec 1.0
# Trial 2: hparam_kl = 0.3, hparam_rec 1.0
# Trial 3: hparam_kl = 0.1, hparam_rec 1.0
# Trial 4: hparam_kl = 0.03, hparam_rec 1.0
# Trial 5,6,7,8: Same as 1,2,3,4 respectively, but with z size 512x8x8 instead of 1024x4x4
# Trial 9: same as 5, except relu instead of leaky relu. (no change, back to relu).
# Trial 10: same as 5, except last_residual_blk has an activation as well. (probably a bad idea)
# Trial 11: same as 5, except last_residual_blk has a IN-RELU-CONV 
# Trial 12: same as 2, with leaky_relu 0.2 
# Trial 13: same as 12, hparam_kl = 0.1, hparam_rec 1.0

# Add Discriminator.
# Trial 14: hparam_kl = 0.1, hparam_rec 1.0, hparam_adv 0.5
# Trial 15: hparam_kl = 0.1, hparam_rec 1.0, hparam_adv 0.1
# Trial 16: hparam_kl = 0.1, hparam_rec 1.0, hparam_adv 0.05

# Trial 17: No discriminator / adversarial, no KL loss, using tanh output instead. hparam_rec 1.0 only loss param.
# Trial 18: same as 17 but with BatchNorm instead of InstanceNorm. no activation at the z.

# Go with the BatchNorm from now on. There were huge problems with InstanceNorm.
# No Discriminator or adv loss or gan loss at this time.

# Trial 19: same as 18, with tanh activation on z.

# Trial 20: no_activation on z, kl loss, hparam_kl = 0.5, hparam_rec 1.0 (good one, will follow this).
# Trial 21: same as 20, hparam_kl = 0.3, hparam_rec 1.0
# Trial 22: same as 21, hparam_kl = 0.1, hparam_rec 1.0
# Trial 23: same as 22, hparam_kl = 0.05, hparam_rec 1.0
# Trial 24: same as 23, hparam_kl = 0.02, hparam_rec 1.0

# From now on, we will use hparam_kl = 0.5, hparam_rec 1.0

# Starting Discriminator
# Trial 25: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 1.0
# Trial 26: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 10.0
# Trial 27: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 30.0

# 25, 26, 27 problem with Gan_loss, male->female, female->male not working, producing same as input.

# Trial 28: Separate encoder for male and female. hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 1.0
# Trial 29: same as 28, disc lr=lr*0.1
# Trial 30: same as 28, disc lr=lr*0.01
# Trial 31: res_blk =1
# Trial 32: adv_loss L2 loss instead of crossentropy (no change).
# Trial 33: back to adv_loss of crossentropy. no KL loss. (no good change).
# Trial 34: hparam_kl = 0.5, hparam_rec 0.01, hparam_adv 1.0
# Trial 35: hparam_kl = 0.01, hparam_rec 0.0, hparam_adv 1.0

# Will try dcgan based model now

####
## All the time we are not reducing adv_loss because of using no.grad() and detach() on the discriminator
####

# Trial 36: model_dcgan_based, only adv_loss 1.0 no other loss, adv lr=lr.
# Trial 37: model.py, only adv_loss 1.0 no other loss, adv lr=lr.
# Trial 38: same as 37, but back to self.res_blk = 4 (good enough)

# Will try with regular model.py, better than dcgan.

# Trial 39: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 1.0
# Trial 40: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.3
# Trial 41: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 3.0
# Trial 42: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.1

# 42 much better than the others (39, 40, 41).

# Trial 43: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.03

# 43 is a bit better than 42

# Will try with common encoder now.

# Trial 44: common encoder for both female and male. hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.1
# Trial 45: common encoder for both female and male. hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.03

# Single encoder (44, 45) gives very similar result as double encoder (42, 43).
# Thus will use single encoder and #45: hparam_adv 0.03

# Starting cycle loss.

# Trial 46: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.03, hparam_cyc 1.0
# Trial 47: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.06, hparam_cyc 1.0

# Both 46 and 47 look very similar, we will go with hparam_adv 0.05, but we are getting a bit high loss on cyc, will increase the hparam_cyc now.

# Trial 48: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.05, hparam_cyc 1.5
# Trial 49: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.05, hparam_cyc 1.5 (same as 48) except:
# with added image flips, also added discFemale(real_male)==false and discMale(real_female)==false

# Just a bit of improvement with 49 than 48. but we are getting a bit high G_adv_loss than 48. increase hparam_adv?

# Trial 50: same as 49, with hparam_adv 0.1
# Trial 51: same as 50, with LSGAN instead of vanilla gan loss. (adv_loss got fixed at 1.000)

# will continue LSGAN with diff param now.

# Trial 52: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.05, hparam_cyc 1.5 (really bad).
# Trial 53: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5

# 53 is the better one between 51, 52, and 53. Will go with 53 now.

# Adding offspirng now.

# Trial 54: hparam_kl = 0.5, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5, have offspring.

# kl_loss getting too high for the offspring adv?

# Trial 55: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5. (slightly better than 54).

# Trial 56: same as 54, have offspring but no offspring_adv_loss.

# Trial 57: hparam_kl 0.0, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5. have offspring, no kl_loss ( too bad).

# Trial 58: only with offspring adv loss, no fake adv loss. param same as 53.

# Trial 59: same as 55, but with two offpsprings by splitting: os1, os2 = split(z1, z2)
# Trial 60: same as 59, but with offspring_loss = ofs_loss_1 + ofs_loss_2 rather than offspring_loss = (ofs_loss_1 + ofs_loss_2) * 0.5 (bad)

# Added patch discriminator.

# Trial 61: same as 59, but with patch discriminator.
# Trial 62: same as 55, but with patch discriminator.

# IMPORTANT: All offspiring were getting saturated at a bad point as we were not training the disc to be it a bad image.
# Will add disc(offspring) == 0 now.

# Trial 63: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5, offspring_flag == "mean".
# Trial 64: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5, offspring_flag == "split"".
# gan_loss a bit higher than the other losses.
# Trial 65: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_flag == "mean".
# Trial 66: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_flag == "split"".

# Probably 65, 66 a bit better than 63, 64.

######################################################################################################
# TO-DO: optimizer vs zero_grad probably wrong, specially check with the cycle consistency part.
# Also, does too many fake and one real to the discriminator change thing?
######################################################################################################

# Starting 128x128 now.

# Trial 67: same as 65, "mean", 128x128.
# Trial 68: same as 66, "split", 128x128.

trial = 68

hparam_kl = 1.0
hparam_rec = 1.0
hparam_adv = 0.5
hparam_cyc = 1.5

device = torch.device("cuda:6")

offspring_flag = "split"
#offspring_flag = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 32
z_dim = (1024, 4, 4)
patch_len = (1, 6 ,6)
num_epochs = 300

lr = 0.0002
beta1 = 0.5

f_data_64x64 = DatasetHDF5("dataset/female1.h5", b_size, True, 100, ['image'])
m_data_64x64 = DatasetHDF5("dataset/male1.h5", b_size, True, 100, ['image'])
disc_f_data_64x64 = DatasetHDF5("dataset/female2.h5", b_size, True, 100, ['image'])
disc_m_data_64x64 = DatasetHDF5("dataset/male2.h5", b_size, True, 100, ['image'])

f_data_128x128 = DatasetHDF5("dataset/female_128x128_1.h5", b_size, True, 100, ['image'])
m_data_128x128 = DatasetHDF5("dataset/male_128x128_1.h5", b_size, True, 100, ['image'])
disc_f_data_128x128 = DatasetHDF5("dataset/female_128x128_2.h5", b_size, True, 100, ['image'])
disc_m_data_128x128 = DatasetHDF5("dataset/male_128x128_2.h5", b_size, True, 100, ['image'])

# custom weights initialization called on Encoder and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

Encoder = model.Encoder().to(device)
Encoder.apply(weights_init)

DecoderFemale = model.Decoder().to(device)
DecoderFemale.apply(weights_init)

DecoderMale = model.Decoder().to(device)
DecoderMale.apply(weights_init)

DiscriminatorFemale = model.Discriminator().to(device)
DiscriminatorFemale.apply(weights_init)

DiscriminatorMale = model.Discriminator().to(device)
DiscriminatorMale.apply(weights_init)

DecoderFemale_128x128 = model.Decoder_128x128().to(device)
DecoderFemale_128x128.apply(weights_init)

DecoderMale_128x128 = model.Decoder_128x128().to(device)
DecoderMale_128x128.apply(weights_init)

DiscriminatorFemale_128x128 = model.Discriminator_128x128().to(device)
DiscriminatorFemale_128x128.apply(weights_init)

DiscriminatorMale_128x128 = model.Discriminator_128x128().to(device)
DiscriminatorMale_128x128.apply(weights_init)

loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()

optimizerGeneratorFemale = optim.Adam(list(Encoder.parameters()) + list(DecoderFemale.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerGeneratorMale = optim.Adam(list(Encoder.parameters()) + list(DecoderMale.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemale = optim.Adam(DiscriminatorFemale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorMale = optim.Adam(DiscriminatorMale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemale_128x128 = optim.Adam(DiscriminatorFemale_128x128.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorMale_128x128 = optim.Adam(DiscriminatorMale_128x128.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDecodeFemale_128x128 = optim.Adam(DecoderFemale_128x128.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDecodeMale_128x128 = optim.Adam(DecoderMale_128x128.parameters(), lr=lr, betas=(beta1, 0.999))

counter_128x128 = 10 # Will run the 128x128 part everything the counter goes zero.
iters = 0
flag = torch.from_numpy(np.array([i%2 for i in xrange(z_dim[0] * z_dim[1] * z_dim[2])]).reshape((z_dim[0], z_dim[1], z_dim[2]))).to(device, torch.float)

print("Starting Training Loop...")
    
log_file = open('log/log_' + str(trial) + '.txt', 'w')
epoch = 1
while True:
    epoch_happened, f_in_img_128x128 = f_data_128x128.load_batch('train')
    _, m_in_img_128x128 = m_data_128x128.load_batch('train')
    _, disc_f_in_img_128x128 = disc_f_data_128x128.load_batch('train')
    _, disc_m_in_img_128x128 = disc_m_data_128x128.load_batch('train')

    _, f_in_img = f_data_64x64.load_batch('train')
    _, m_in_img = m_data_64x64.load_batch('train')
    _, disc_f_in_img = disc_f_data_64x64.load_batch('train')
    _, disc_m_in_img = disc_m_data_64x64.load_batch('train')

    if f_in_img.shape[0] != b_size or m_in_img.shape[0] != b_size:
        continue
    if disc_f_in_img.shape[0] != b_size or disc_m_in_img.shape[0] != b_size:
        continue
    if f_in_img_128x128.shape[0] != b_size or m_in_img_128x128.shape[0] != b_size:
        continue
    if disc_f_in_img_128x128.shape[0] != b_size or disc_m_in_img_128x128.shape[0] != b_size:
        continue

    for i in xrange(b_size):
        if np.random.random() > 0.5:
            f_in_img[i] = np.fliplr(f_in_img[i])
            f_in_img_128x128[i] = np.fliplr(f_in_img_128x128[i])
        if np.random.random() > 0.5:
            m_in_img[i] = np.fliplr(m_in_img[i])
            m_in_img_128x128[i] = np.fliplr(m_in_img_128x128[i])
        if np.random.random() > 0.5:
            disc_f_in_img[i] = np.fliplr(disc_f_in_img[i])
        if np.random.random() > 0.5:
            disc_m_in_img[i] = np.fliplr(disc_m_in_img[i])

    disc_f_in_img = preprocess_convert_images(disc_f_in_img).to(device)
    disc_m_in_img = preprocess_convert_images(disc_m_in_img).to(device)
    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)
    
    disc_f_in_img_128x128 = preprocess_convert_images(disc_f_in_img_128x128).to(device)
    disc_m_in_img_128x128 = preprocess_convert_images(disc_m_in_img_128x128).to(device)
    f_in_img_128x128 = preprocess_convert_images(f_in_img_128x128).to(device)
    m_in_img_128x128 = preprocess_convert_images(m_in_img_128x128).to(device)

    zero_label_flat = torch.full((b_size,), 0, device=device)    
    zero_label = torch.full((b_size, patch_len[0], patch_len[1], patch_len[2]), 0, device=device)    
    one_label = torch.full((b_size, patch_len[0], patch_len[1], patch_len[2]), 1, device=device)    

    # Female Discriminator loss.
    DiscriminatorFemale.zero_grad()
    disc_f_out = DiscriminatorFemale(disc_f_in_img)
    loss_real_D_female = loss_mse(disc_f_out, one_label)
    loss_real_D_female.backward()

    with torch.no_grad():
        z_male = Encoder(disc_m_in_img)
        fake_f_out_img, _ = DecoderFemale(z_male)
        fake_f_out_img = fake_f_out_img.detach()
    disc_f_out = DiscriminatorFemale(fake_f_out_img)
    disc_m_out = DiscriminatorFemale(disc_m_in_img)

    if offspring_flag == "mean":
        with torch.no_grad():
            z_female = Encoder(disc_f_in_img)
            z_offspring = (z_female + z_male) / 2.0
            offspring_f_out_img, _ = DecoderFemale(z_offspring)
        disc_offspring_f_out = DiscriminatorFemale(offspring_f_out_img)
    else:
        with torch.no_grad():
            z_female = Encoder(disc_f_in_img)
            z_offspring1 = z_female * flag + z_male * (1.0 - flag)
            z_offspring2 = z_female * (1.0 - flag) + z_male * flag
            offspring_f_out_img1, _ = DecoderFemale(z_offspring1)
            offspring_f_out_img2, _ = DecoderFemale(z_offspring2)
        disc_offspring_f_out1 = DiscriminatorFemale(offspring_f_out_img1)
        disc_offspring_f_out2 = DiscriminatorFemale(offspring_f_out_img2)

    if offspring_flag == "mean":
        loss_fake_D_female = loss_mse(torch.cat((disc_f_out, disc_m_out, disc_offspring_f_out), 0), torch.cat((zero_label, zero_label, zero_label), 0))
    else:
        loss_fake_D_female = loss_mse(torch.cat((disc_f_out, disc_m_out, disc_offspring_f_out1, disc_offspring_f_out2), 0), torch.cat((zero_label, zero_label, zero_label, zero_label), 0))
    loss_fake_D_female.backward()

    optimizerDiscriminatorFemale.step()

    # Male Discriminator loss.
    DiscriminatorMale.zero_grad()
    disc_m_out = DiscriminatorMale(disc_m_in_img)
    loss_real_D_male = loss_mse(disc_m_out, one_label)
    loss_real_D_male.backward()

    with torch.no_grad():
        z_female = Encoder(disc_f_in_img)
        fake_m_out_img, _ = DecoderMale(z_female)
        fake_m_out_img = fake_m_out_img.detach()
    disc_m_out = DiscriminatorMale(fake_m_out_img)
    disc_f_out = DiscriminatorMale(disc_f_in_img)

    if offspring_flag == "mean":
        with torch.no_grad():
            z_male = Encoder(disc_m_in_img)
            z_offspring = (z_female + z_male) / 2.0
            offspring_m_out_img, _ = DecoderMale(z_offspring)
        disc_offspring_m_out = DiscriminatorMale(offspring_m_out_img)
    else:
        with torch.no_grad():
            z_male = Encoder(disc_m_in_img)
            z_offspring1 = z_female * flag + z_male * (1.0 - flag)
            z_offspring2 = z_female * (1.0 - flag) + z_male * flag
            offspring_m_out_img1, _ = DecoderMale(z_offspring1)
            offspring_m_out_img2, _ = DecoderMale(z_offspring2)
        disc_offspring_m_out1 = DiscriminatorMale(offspring_m_out_img1)
        disc_offspring_m_out2 = DiscriminatorMale(offspring_m_out_img2)

    if offspring_flag == "mean":
        loss_fake_D_male = loss_mse(torch.cat((disc_m_out, disc_f_out, disc_offspring_m_out), 0), torch.cat((zero_label, zero_label, zero_label), 0))
    else:
        loss_fake_D_male = loss_mse(torch.cat((disc_m_out, disc_f_out, disc_offspring_m_out1, disc_offspring_m_out2), 0), torch.cat((zero_label, zero_label, zero_label, zero_label), 0))
    loss_fake_D_male.backward()

    optimizerDiscriminatorMale.step()

    # Female KL, Reconstruction, regular Adversarial, Cycle, and Offspring Adversarial loss.
    Encoder.zero_grad()
    DecoderFemale.zero_grad()
    DecoderMale.zero_grad()
    z_female = Encoder(f_in_img)
    f_out_img, _ = DecoderFemale(z_female)

    loss_rec_female = loss_l1(f_out_img, f_in_img)
    z_std, z_mean = torch.std_mean(z_female, dim=(1,2,3))
    loss_kl_female = loss_l1(z_mean * z_mean + z_std - torch.log(z_std) - 1.0, zero_label_flat)

    z_male = Encoder(m_in_img)
    fake_f_out_img, inter_fake_f_out = DecoderFemale(z_male)
    disc_f_out = DiscriminatorFemale(fake_f_out_img)
    loss_G_female = loss_mse(disc_f_out, one_label)

    cyc_f_out_img, _ = DecoderFemale(Encoder(DecoderMale(z_female)[0]))
    loss_cyc_female = loss_l1(cyc_f_out_img, f_in_img)

    if offspring_flag == "mean":
        z_offspring = (z_female + z_male) / 2.0
        offspring_f_out_img, inter_os_f_out = DecoderFemale(z_offspring)
        disc_offspring_f_out = DiscriminatorFemale(offspring_f_out_img)
        loss_G_offspring_female = loss_mse(disc_offspring_f_out, one_label)
    else:
        z_offspring1 = z_female * flag + z_male * (1.0 - flag)
        z_offspring2 = z_female * (1.0 - flag) + z_male * flag
        offspring_f_out_img1, inter_os_f_out1 = DecoderFemale(z_offspring1)
        offspring_f_out_img2, inter_os_f_out2 = DecoderFemale(z_offspring2)
        disc_offspring_f_out1 = DiscriminatorFemale(offspring_f_out_img1)
        disc_offspring_f_out2 = DiscriminatorFemale(offspring_f_out_img2)
        loss_G_offspring_female = (loss_mse(disc_offspring_f_out1, one_label) + loss_mse(disc_offspring_f_out2, one_label)) * 0.5

    loss_total_female = hparam_rec * loss_rec_female + hparam_kl * loss_kl_female + hparam_adv * (loss_G_female + loss_G_offspring_female) + hparam_cyc * loss_cyc_female
    loss_total_female.backward()

    optimizerGeneratorFemale.step()

    # Male KL, Reconstruction, regular Adversarial, Cycle, and Offspring Adversarial loss.
    Encoder.zero_grad()
    DecoderFemale.zero_grad()
    DecoderMale.zero_grad()
    z_male = Encoder(m_in_img)
    m_out_img, _ = DecoderMale(z_male)

    loss_rec_male = loss_l1(m_out_img, m_in_img)
    z_std, z_mean = torch.std_mean(z_male, dim=(1,2,3))
    loss_kl_male = loss_l1(z_mean * z_mean + z_std - torch.log(z_std) - 1.0, zero_label_flat)

    z_female = Encoder(f_in_img)
    fake_m_out_img, inter_fake_m_out = DecoderMale(z_female)
    disc_m_out = DiscriminatorMale(fake_m_out_img)
    loss_G_male = loss_mse(disc_m_out, one_label)

    cyc_m_out_img, _ = DecoderMale(Encoder(DecoderFemale(z_male)[0]))
    loss_cyc_male = loss_l1(cyc_m_out_img, m_in_img)

    if offspring_flag == "mean":
        z_offspring = (z_female + z_male) / 2.0
        offspring_m_out_img, inter_os_m_out = DecoderMale(z_offspring)
        disc_offspring_m_out = DiscriminatorMale(offspring_m_out_img)
        loss_G_offspring_male = loss_mse(disc_offspring_m_out, one_label)
    else:
        z_offspring1 = z_female * flag + z_male * (1.0 - flag)
        z_offspring2 = z_female * (1.0 - flag) + z_male * flag
        offspring_m_out_img1, inter_os_m_out1 = DecoderMale(z_offspring1)
        offspring_m_out_img2, inter_os_m_out2 = DecoderMale(z_offspring2)
        disc_offspring_m_out1 = DiscriminatorMale(offspring_m_out_img1)
        disc_offspring_m_out2 = DiscriminatorMale(offspring_m_out_img2)
        loss_G_offspring_male = (loss_mse(disc_offspring_m_out1, one_label) + loss_mse(disc_offspring_m_out2, one_label)) * 0.5

    loss_total_male = hparam_rec * loss_rec_male + hparam_kl * loss_kl_male + hparam_adv * (loss_G_male + loss_G_offspring_male) + hparam_cyc * loss_cyc_male
    loss_total_male.backward()

    optimizerGeneratorMale.step()

    loss_total = loss_total_female + loss_total_male

    # Run the 128x128 part
    counter_128x128 = counter_128x128 - 1
    if epoch_happened or counter_128x128 == 0:
        counter_128x128 = max(1, 10 - int(epoch / 5)) # starts with 10, and goes down 1 every 5 epoch, until it goes down to 1 after 45 epochs.

        # Female Discriminator loss.
        DiscriminatorFemale_128x128.zero_grad()
        disc_f_out = DiscriminatorFemale_128x128(disc_f_in_img_128x128)
        loss_real_D_female_128x128 = loss_mse(disc_f_out, one_label)
        loss_real_D_female_128x128.backward()

        with torch.no_grad():
            fake_f_out_img_128x128 = DecoderFemale_128x128(inter_fake_f_out).detach()
        disc_fake_f_out = DiscriminatorFemale_128x128(fake_f_out_img_128x128)
        disc_m_out = DiscriminatorFemale_128x128(disc_m_in_img_128x128)

        if offspring_flag == "mean":
            with torch.no_grad():
                os_f_out_img_128x128 = DecoderFemale_128x128(inter_os_f_out).detach()
            disc_os_f_out = DiscriminatorFemale_128x128(os_f_out_img_128x128)
        else:
            with torch.no_grad():
                os_f_out_img1_128x128 = DecoderFemale_128x128(inter_os_f_out1).detach()
                os_f_out_img2_128x128 = DecoderFemale_128x128(inter_os_f_out2).detach()
            disc_os_f_out1 = DiscriminatorFemale_128x128(os_f_out_img1_128x128)
            disc_os_f_out2 = DiscriminatorFemale_128x128(os_f_out_img2_128x128)
           
        if offspring_flag == "mean":
            loss_fake_D_female_128x128 = loss_mse(torch.cat((disc_fake_f_out, disc_m_out, disc_os_f_out), 0), torch.cat((zero_label, zero_label, zero_label), 0))
        else:
            loss_fake_D_female_128x128 = loss_mse(torch.cat((disc_fake_f_out, disc_m_out, disc_os_f_out1, disc_os_f_out2), 0), torch.cat((zero_label, zero_label, zero_label, zero_label), 0))
        loss_fake_D_female_128x128.backward()

        optimizerDiscriminatorFemale_128x128.step()

        # Male Discriminator loss.
        DiscriminatorMale_128x128.zero_grad()
        disc_m_out = DiscriminatorMale_128x128(disc_m_in_img_128x128)
        loss_real_D_male_128x128 = loss_mse(disc_m_out, one_label)
        loss_real_D_male_128x128.backward()

        with torch.no_grad():
            fake_m_out_img_128x128 = DecoderMale_128x128(inter_fake_m_out).detach()
        disc_fake_m_out = DiscriminatorMale_128x128(fake_m_out_img_128x128)
        disc_f_out = DiscriminatorMale_128x128(disc_f_in_img_128x128)

        if offspring_flag == "mean":
            with torch.no_grad():
                os_m_out_img_128x128 = DecoderMale_128x128(inter_os_m_out).detach()
            disc_os_m_out = DiscriminatorMale_128x128(os_m_out_img_128x128)
        else:
            with torch.no_grad():
                os_m_out_img1_128x128 = DecoderMale_128x128(inter_os_m_out1).detach()
                os_m_out_img2_128x128 = DecoderMale_128x128(inter_os_m_out2).detach()
            disc_os_m_out1 = DiscriminatorMale_128x128(os_m_out_img1_128x128)
            disc_os_m_out2 = DiscriminatorMale_128x128(os_m_out_img2_128x128)
           
        if offspring_flag == "mean":
            loss_fake_D_male_128x128 = loss_mse(torch.cat((disc_fake_m_out, disc_f_out, disc_os_m_out), 0), torch.cat((zero_label, zero_label, zero_label), 0))
        else:
            loss_fake_D_male_128x128 = loss_mse(torch.cat((disc_fake_m_out, disc_f_out, disc_os_m_out1, disc_os_m_out2), 0), torch.cat((zero_label, zero_label, zero_label, zero_label), 0))
        loss_fake_D_male_128x128.backward()

        optimizerDiscriminatorMale_128x128.step()

        # Female Adversarial loss.
        DecoderFemale_128x128.zero_grad()

        fake_f_out_img_128x128 = DecoderFemale_128x128(inter_fake_f_out.detach())
        disc_fake_f_out = DiscriminatorFemale_128x128(fake_f_out_img_128x128)

        if offspring_flag == "mean":
            os_f_out_img_128x128 = DecoderFemale_128x128(inter_os_f_out.detach())
            disc_os_f_out = DiscriminatorFemale_128x128(os_f_out_img_128x128)
            loss_G_female_128x128 = loss_mse(torch.cat((disc_fake_f_out, disc_os_f_out), 0), torch.cat((one_label, one_label), 0))
        else:
            os_f_out_img1_128x128 = DecoderFemale_128x128(inter_os_f_out1.detach())
            os_f_out_img2_128x128 = DecoderFemale_128x128(inter_os_f_out2.detach())
            disc_os_f_out1 = DiscriminatorFemale_128x128(os_f_out_img1_128x128)
            disc_os_f_out2 = DiscriminatorFemale_128x128(os_f_out_img2_128x128)
            loss_G_female_128x128 = loss_mse(torch.cat((disc_fake_f_out, disc_os_f_out1, disc_os_f_out2), 0), torch.cat((one_label, one_label, one_label), 0))

        loss_G_female_128x128.backward()
        optimizerDecodeFemale_128x128.step()

        # Male Adversarial loss.
        DecoderMale_128x128.zero_grad()

        fake_m_out_img_128x128 = DecoderMale_128x128(inter_fake_m_out.detach())
        disc_fake_m_out = DiscriminatorMale_128x128(fake_m_out_img_128x128)

        if offspring_flag == "mean":
            os_m_out_img_128x128 = DecoderMale_128x128(inter_os_m_out.detach())
            disc_os_m_out = DiscriminatorMale_128x128(os_m_out_img_128x128)
            loss_G_male_128x128 = loss_mse(torch.cat((disc_fake_m_out, disc_os_m_out), 0), torch.cat((one_label, one_label), 0))
        else:
            os_m_out_img1_128x128 = DecoderMale_128x128(inter_os_m_out1.detach())
            os_m_out_img2_128x128 = DecoderMale_128x128(inter_os_m_out2.detach())
            disc_os_m_out1 = DiscriminatorMale_128x128(os_m_out_img1_128x128)
            disc_os_m_out2 = DiscriminatorMale_128x128(os_m_out_img2_128x128)
            loss_G_male_128x128 = loss_mse(torch.cat((disc_fake_m_out, disc_os_m_out1, disc_os_m_out2), 0), torch.cat((one_label, one_label, one_label), 0))

        loss_G_male_128x128.backward()
        optimizerDecodeMale_128x128.step()

    # Output training stats
    if iters > 0 and iters % 50 == 0:
        print_str = '[trial: %d][iter: %d][epoch: %d/%d]\tL_Total: %.3f\nL_KL: %.3f/%.3f\tL_Rec: %.3f/%.3f\tL_Cyc: %.3f/%.3f\nL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f\tL_Adv_OS: %.3f/%.3f\n128x128:\tL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f' % (trial, iters, epoch, num_epochs, loss_total.item(), loss_kl_female.item(), loss_kl_male.item(), loss_rec_female.item(), loss_rec_male.item(), loss_cyc_female.item(), loss_cyc_male.item(), loss_real_D_female.item(), loss_real_D_male.item(), loss_fake_D_female.item(), loss_fake_D_male.item(), loss_G_female.item(), loss_G_male.item(), loss_G_offspring_female.item(), loss_G_offspring_male.item(), loss_real_D_female_128x128.item(), loss_real_D_male_128x128.item(), loss_fake_D_female_128x128.item(), loss_fake_D_male_128x128.item(), loss_G_female_128x128.item(), loss_G_male_128x128.item())
        print(print_str)
        log_file.write(print_str + '\n')
        log_file.flush()
    
    iters += 1
    if epoch_happened == True:
    #if iters > 1 and iters % 50 == 1:
        n_row = 8
        n_col = 1
        n_col_cell = -1
        f_in_img = postprocess_convert_images(f_in_img.cpu())
        m_in_img = postprocess_convert_images(m_in_img.cpu())
        f_out_img = postprocess_convert_images(f_out_img.detach().cpu())
        m_out_img = postprocess_convert_images(m_out_img.detach().cpu())
        fake_f_out_img = postprocess_convert_images(fake_f_out_img.detach().cpu())
        fake_m_out_img = postprocess_convert_images(fake_m_out_img.detach().cpu())
        fake_f_out_img_128x128 = postprocess_convert_images(fake_f_out_img_128x128.detach().cpu())
        fake_m_out_img_128x128 = postprocess_convert_images(fake_m_out_img_128x128.detach().cpu())
        if offspring_flag == "mean": 
            offspring_f_out_img1 = postprocess_convert_images(offspring_f_out_img.detach().cpu())
            offspring_m_out_img1 = postprocess_convert_images(offspring_m_out_img.detach().cpu())
            os_f_out_img1_128x128 = postprocess_convert_images(os_f_out_img_128x128.detach().cpu())
            os_m_out_img1_128x128 = postprocess_convert_images(os_m_out_img_128x128.detach().cpu())
            n_col_cell = 12
        else:
            offspring_f_out_img1 = postprocess_convert_images(offspring_f_out_img1.detach().cpu())
            offspring_f_out_img2 = postprocess_convert_images(offspring_f_out_img2.detach().cpu())
            offspring_m_out_img1 = postprocess_convert_images(offspring_m_out_img1.detach().cpu())
            offspring_m_out_img2 = postprocess_convert_images(offspring_m_out_img2.detach().cpu())
            os_f_out_img1_128x128 = postprocess_convert_images(os_f_out_img1_128x128.detach().cpu())
            os_f_out_img2_128x128 = postprocess_convert_images(os_f_out_img2_128x128.detach().cpu())
            os_m_out_img1_128x128 = postprocess_convert_images(os_m_out_img1_128x128.detach().cpu())
            os_m_out_img2_128x128 = postprocess_convert_images(os_m_out_img2_128x128.detach().cpu())
            n_col_cell = 16
        cyc_f_out_img = postprocess_convert_images(cyc_f_out_img.detach().cpu())
        cyc_m_out_img = postprocess_convert_images(cyc_m_out_img.detach().cpu())
        img_row = 128
        img_col = 128
        image = np.zeros((img_row*n_row, img_col*n_col*n_col_cell, 3), dtype=np.uint8)
        for r in xrange(n_row):
            for c in xrange(0, n_col*n_col_cell, n_col_cell):
                image[r*img_row:(r+1)*img_row-img_row/2, c*img_col:(c+1)*img_col-img_col/2, :] = f_in_img[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+1)*img_col:(c+2)*img_col-img_col/2, :] = m_in_img[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+2)*img_col:(c+3)*img_col-img_col/2, :] = f_out_img[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+3)*img_col:(c+4)*img_col-img_col/2, :] = m_out_img[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+4)*img_col:(c+5)*img_col-img_col/2, :] = fake_f_out_img[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+5)*img_col:(c+6)*img_col-img_col/2, :] = fake_m_out_img[r*n_col+c//n_col_cell]
                #image[r*img_row:(r+1)*img_row, (c+6)*img_col:(c+7)*img_col, :] = cyc_f_out_img[r*n_col+c//n_col_cell]
                #image[r*img_row:(r+1)*img_row, (c+7)*img_col:(c+8)*img_col, :] = cyc_m_out_img[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+6)*img_col:(c+7)*img_col-img_col/2, :] = offspring_f_out_img1[r*n_col+c//n_col_cell]
                image[r*img_row:(r+1)*img_row-img_row/2, (c+7)*img_col:(c+8)*img_col-img_col/2, :] = offspring_m_out_img1[r*n_col+c//n_col_cell]
                if n_col_cell == 12:
                    image[r*img_row:(r+1)*img_row, (c+8)*img_col:(c+9)*img_col, :] = fake_f_out_img_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+9)*img_col:(c+10)*img_col, :] = fake_m_out_img_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+10)*img_col:(c+11)*img_col, :] = os_f_out_img1_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+11)*img_col:(c+12)*img_col, :] = os_m_out_img1_128x128[r*n_col+c//n_col_cell]
                else:
                    image[r*img_row:(r+1)*img_row-img_row/2, (c+8)*img_col:(c+9)*img_col-img_col/2, :] = offspring_f_out_img2[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row-img_row/2, (c+9)*img_col:(c+10)*img_col-img_col/2, :] = offspring_m_out_img2[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+10)*img_col:(c+11)*img_col, :] = fake_f_out_img_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+11)*img_col:(c+12)*img_col, :] = fake_m_out_img_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+12)*img_col:(c+13)*img_col, :] = os_f_out_img1_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+13)*img_col:(c+14)*img_col, :] = os_m_out_img1_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+14)*img_col:(c+15)*img_col, :] = os_f_out_img2_128x128[r*n_col+c//n_col_cell]
                    image[r*img_row:(r+1)*img_row, (c+15)*img_col:(c+16)*img_col, :] = os_m_out_img2_128x128[r*n_col+c//n_col_cell]
        imsave('sample/train_' + str(trial) + "_" + str(epoch) + '.jpg', image)    
        if epoch % 5 == 0:
            torch.save(Encoder.state_dict(), 'weight/encoder_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DecoderFemale.state_dict(), 'weight/decoder_f_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DecoderMale.state_dict(), 'weight/decoder_m_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DiscriminatorFemale.state_dict(), 'weight/discriminator_f_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DiscriminatorMale.state_dict(), 'weight/discriminator_m_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DecoderFemale_128x128.state_dict(), 'weight/decoder_128x128_f_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DecoderMale_128x128.state_dict(), 'weight/decoder_128x128_m_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DiscriminatorFemale_128x128.state_dict(), 'weight/discriminator_128x128_f_' + str(trial) + '_' + str(epoch) + '.pt')
            torch.save(DiscriminatorMale_128x128.state_dict(), 'weight/discriminator_128x128_m_' + str(trial) + '_' + str(epoch) + '.pt')
        epoch += 1
        if epoch > num_epochs:
            break

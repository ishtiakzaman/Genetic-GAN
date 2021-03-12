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
import model
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover

#########################################
# CONTINUING FROM train.py with trial 69
#########################################

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

# Trial 63: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 64: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.2, hparam_cyc 1.5, offspring_mode == "split"".
# gan_loss a bit higher than the other losses.
# Trial 65: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 66: hparam_kl = 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_mode == "split"".

# Probably 65, 66 a bit better than 63, 64.

######################################################################################################
# TO-DO: optimizer vs zero_grad probably wrong, specially check with the cycle consistency part.
# Also, does too many fake and one real to the discriminator change thing?
######################################################################################################

# Starting 128x128 now with batch size 32 instead of 64 (memory constraints).

# Trial 67: same as 65, "mean", 128x128.
# Trial 68: same as 66, "split", 128x128.

# 67, 68 128x128 are real bad, 64x64 part is still good. may be be a bit better than 65 and 66, is this because of batch size 32 instead of 64?

#########################################
# CONTINUING FROM train.py with trial 69
#########################################

# Trial 69: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 70: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 0.25, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 71: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 1.0, hparam_cyc 1.5, offspring_mode == "mean".

# Trial 72: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_mode == "split".
# Trial 73: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 0.25, hparam_cyc 1.5, offspring_mode == "split".
# Trial 74: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 1.0, hparam_cyc 1.5, offspring_mode == "split".

# Trial 75: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 0.5, hparam_cyc 1.5, offspring_mode == "part".
# Trial 76: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 0.25, hparam_cyc 1.5, offspring_mode == "part".
# Trial 77: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 1.0, hparam_cyc 1.5, offspring_mode == "part".

# Looks like hparam_adv 1.0 is the best for all cases. will try even higher now.

# Trial 78: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 2.0, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 79: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "mean".

# Trial 80: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 2.0, hparam_cyc 1.5, offspring_mode == "split".
# Trial 81: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "split".

# Trial 82: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 2.0, hparam_cyc 1.5, offspring_mode == "part".
# Trial 83: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "part".

# Some acceptable results:
# 79_235, 79_171, 79_98, 79_95, 79_93, 79_92, 79_45, 79_25
# 78_59, 78_60
# 80_50, 80_53, 80_58, 80_142
# 81_58
# 82_7, 82_41, 82_43, 82_44, 82_46, 82_47, 82_176
# 83_32, 83_40, 83_45, 83_48, 83_53, 83_112, 83_126, 83_127, 83_130, 83_131

# Trial 84: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 8.0, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 85: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 8.0, hparam_cyc 1.5, offspring_mode == "split".
# Trial 86: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 8.0, hparam_cyc 1.5, offspring_mode == "part".

# We prefer hparam_adv 4.0

# Trial 87: same as 79, except we are not using disc_fm_in_img in the disc false, using f/m_in_img instead.
# Trial 88: same as 87, but with "split"
# Trial 89: same as 81, but with "part"

trial = 89

hparam_kl = 1.0
hparam_rec = 1.0
hparam_adv = 4.0
hparam_cyc = 1.5

device = torch.device("cuda:6")

offspring_mode = "part"
#offspring_mode = "split"
#offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 32
z_dim = (1024, 4, 3)
patch_len = (1, 4 ,3)
num_epochs = 100

lr = 0.0002
beta1 = 0.5

f_data = DatasetHDF5("dataset/female_low_1.h5", b_size, True, 100, ['image'])
m_data = DatasetHDF5("dataset/male_low_1.h5", b_size, True, 100, ['image'])
disc_f_data = DatasetHDF5("dataset/female_low_2.h5", b_size, True, 100, ['image'])
disc_m_data = DatasetHDF5("dataset/male_low_2.h5", b_size, True, 100, ['image'])

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

loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()

optimizerGenerator = optim.Adam(list(Encoder.parameters()) + list(DecoderFemale.parameters()) + list(DecoderMale.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemale = optim.Adam(DiscriminatorFemale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorMale = optim.Adam(DiscriminatorMale.parameters(), lr=lr, betas=(beta1, 0.999))

iters = 0

print("Starting Training Loop...")
    
log_file = open('log/log_' + str(trial) + '.txt', 'w')
graph_file = open('graph/graph_' + str(trial) + '.csv', 'w')
graph_file.write('iters,total,kl,rec,cyc,real,fake,G\n')
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

    for i in xrange(b_size):
        if np.random.random() > 0.5:
            f_in_img[i] = np.fliplr(f_in_img[i])
        if np.random.random() > 0.5:
            m_in_img[i] = np.fliplr(m_in_img[i])
        if np.random.random() > 0.5:
            disc_f_in_img[i] = np.fliplr(disc_f_in_img[i])
        if np.random.random() > 0.5:
            disc_m_in_img[i] = np.fliplr(disc_m_in_img[i])

    disc_f_in_img = preprocess_convert_images(disc_f_in_img).to(device)
    disc_m_in_img = preprocess_convert_images(disc_m_in_img).to(device)
    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    zero_label_flat = torch.full((b_size,), 0, device=device)    
    zero_label = torch.full((b_size, patch_len[0], patch_len[1], patch_len[2]), 0, device=device)    
    one_label = torch.full((b_size, patch_len[0], patch_len[1], patch_len[2]), 1, device=device)    

    # Female Discriminator loss.
    DiscriminatorFemale.zero_grad()
    disc_f_out = DiscriminatorFemale(disc_f_in_img)
    loss_real_D_female = loss_mse(disc_f_out, one_label)
    loss_real_D_female.backward()

    disc_m_out = DiscriminatorFemale(disc_m_in_img)

    with torch.no_grad():
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring).detach()
    disc_offspring_f_out = DiscriminatorFemale(offspring_f_out_img)

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
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        offspring_m_out_img = DecoderMale(z_offspring).detach()
    disc_offspring_m_out = DiscriminatorMale(offspring_m_out_img)

    loss_fake_D_male = loss_mse(torch.cat((disc_f_out, disc_offspring_m_out), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_male.backward()

    optimizerDiscriminatorMale.step()

    # Female KL, Reconstruction, regular Adversarial, Cycle, and Offspring Adversarial loss.
    Encoder.zero_grad()
    DecoderFemale.zero_grad()
    DecoderMale.zero_grad()
    z_female = Encoder(f_in_img)
    f_out_img = DecoderFemale(z_female)

    loss_rec_female = loss_l1(f_out_img, f_in_img)
    z_std, z_mean = torch.std_mean(z_female, dim=(1,2,3))
    loss_kl_female = loss_l1(z_mean * z_mean + z_std - torch.log(z_std) - 1.0, zero_label_flat)

    z_male = Encoder(m_in_img)
    z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
    offspring_f_out_img = DecoderFemale(z_offspring)
    disc_offspring_f_out = DiscriminatorFemale(offspring_f_out_img)
    loss_G_female = loss_mse(disc_offspring_f_out, one_label)

    cyc_f_out_img = DecoderFemale(Encoder(DecoderMale(z_female)))
    loss_cyc_female = loss_l1(cyc_f_out_img, f_in_img)

    loss_total_female = hparam_rec * loss_rec_female + hparam_kl * loss_kl_female + hparam_adv * loss_G_female + hparam_cyc * loss_cyc_female
    loss_total_female.backward()

    optimizerGenerator.step()

    # Male KL, Reconstruction, regular Adversarial, Cycle, and Offspring Adversarial loss.
    Encoder.zero_grad()
    DecoderFemale.zero_grad()
    DecoderMale.zero_grad()
    z_male = Encoder(m_in_img)
    m_out_img = DecoderMale(z_male)

    loss_rec_male = loss_l1(m_out_img, m_in_img)
    z_std, z_mean = torch.std_mean(z_male, dim=(1,2,3))
    loss_kl_male = loss_l1(z_mean * z_mean + z_std - torch.log(z_std) - 1.0, zero_label_flat)

    z_female = Encoder(f_in_img)
    z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
    offspring_m_out_img = DecoderMale(z_offspring)
    disc_offspring_m_out = DiscriminatorMale(offspring_m_out_img)
    loss_G_male = loss_mse(disc_offspring_m_out, one_label)

    cyc_m_out_img = DecoderMale(Encoder(DecoderFemale(z_male)))
    loss_cyc_male = loss_l1(cyc_m_out_img, m_in_img)

    loss_total_male = hparam_rec * loss_rec_male + hparam_kl * loss_kl_male + hparam_adv * loss_G_male + hparam_cyc * loss_cyc_male
    loss_total_male.backward()

    optimizerGenerator.step()

    loss_total = loss_total_female + loss_total_male
      
    # Output training stats
    if iters % 50 == 0:
        print_str = '[trial: %d][iter: %d][epoch: %d/%d]\tL_Total: %.3f\nL_KL: %.3f/%.3f\tL_Rec: %.3f/%.3f\tL_Cyc: %.3f/%.3f\nL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f' % (trial, iters, epoch, num_epochs, loss_total.item(), loss_kl_female.item(), loss_kl_male.item(), loss_rec_female.item(), loss_rec_male.item(), loss_cyc_female.item(), loss_cyc_male.item(), loss_real_D_female.item(), loss_real_D_male.item(), loss_fake_D_female.item(), loss_fake_D_male.item(), loss_G_female.item(), loss_G_male.item())
        print(print_str)
        log_file.write(print_str + '\n')
        log_file.flush()
        graph_str = str(iters) + ',' + str(loss_total.item()) + ',' + str((loss_kl_female.item() + loss_kl_male.item()) * 0.5) + ',' + str((loss_rec_female.item() + loss_rec_male.item()) * 0.5) + ',' + str((loss_cyc_female.item() + loss_cyc_male.item()) * 0.5) + ',' + str((loss_real_D_female.item() + loss_real_D_male.item()) * 0.5) + ',' + str((loss_fake_D_female.item() + loss_fake_D_male.item()) * 0.5) + ',' + str((loss_G_female.item() + loss_G_male.item()) * 0.5)
        graph_file.write(graph_str + '\n')
        graph_file.flush()
    
    iters += 1
    child_index = (child_index + 1) % 5
    if epoch_happened == True:
    #if iters > 1 and iters % 50 == 1:
        n_row = 10
        n_col = 16
        f_in_img = postprocess_convert_images(f_in_img.cpu())
        m_in_img = postprocess_convert_images(m_in_img.cpu())
        f_out_img = postprocess_convert_images(f_out_img.detach().cpu())
        m_out_img = postprocess_convert_images(m_out_img.detach().cpu())
        cyc_f_out_img = postprocess_convert_images(cyc_f_out_img.detach().cpu())
        cyc_m_out_img = postprocess_convert_images(cyc_m_out_img.detach().cpu())
        offspring_female_list = []
        offspring_male_list = []
        for ci in xrange(5):
            offspring_female_list.append(postprocess_convert_images(DecoderFemale(crossover(z_female, z_male, ci, offspring_mode, device)).detach().cpu()))
            offspring_male_list.append(postprocess_convert_images(DecoderMale(crossover(z_female, z_male, ci, offspring_mode, device)).detach().cpu()))
        img_row = 55
        img_col = 45
        image = np.zeros((img_row*n_row, img_col*n_col, 3), dtype=np.uint8)
        for r in xrange(n_row):
            image[r*img_row:(r+1)*img_row, img_col*0:img_col*1, :] = f_in_img[r]
            image[r*img_row:(r+1)*img_row, img_col*1:img_col*2, :] = m_in_img[r]
            image[r*img_row:(r+1)*img_row, img_col*2:img_col*3, :] = f_out_img[r]
            image[r*img_row:(r+1)*img_row, img_col*3:img_col*4, :] = m_out_img[r]
            image[r*img_row:(r+1)*img_row, img_col*4:img_col*5, :] = cyc_f_out_img[r]
            image[r*img_row:(r+1)*img_row, img_col*5:img_col*6, :] = cyc_m_out_img[r]
            for ci in xrange(5):
                image[r*img_row:(r+1)*img_row, img_col*(6+ci):img_col*(7+ci), :] = offspring_female_list[ci][r]
            for ci in xrange(5):
                image[r*img_row:(r+1)*img_row, img_col*(11+ci):img_col*(12+ci), :] = offspring_male_list[ci][r]
        imsave('sample_low/train_' + str(trial) + "_" + str(epoch) + '.jpg', image)    
        #if epoch % 5 == 0:
        torch.save(Encoder.state_dict(), 'weight/encoder_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DecoderFemale.state_dict(), 'weight/decoder_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DecoderMale.state_dict(), 'weight/decoder_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorFemale.state_dict(), 'weight/discriminator_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorMale.state_dict(), 'weight/discriminator_m_' + str(trial) + '_' + str(epoch) + '.pt')
        epoch += 1
        if epoch > num_epochs:
            break

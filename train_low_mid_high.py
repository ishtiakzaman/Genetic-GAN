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

# Trials from train_low was:
# Trial 79: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 81: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "split".
# Trial 83: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "part".

# Trial 1: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "mean".
# Trial 2: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "split".
# Trial 3: hparam_kl 1.0, hparam_rec 1.0, hparam_adv 4.0, hparam_cyc 1.5, offspring_mode == "part".
# Trial 4: same as 1, except mid backpropagation propagate low part also.

# Trials from low_mid was:
# 1-4 had some error in the show figure part.

# Trial 5: same as 4, except mid discriminator lr = 0.0002 * 0.3
# Trial 6: same as 4, except mid discriminator lr = 0.0002 * 0.2
# Trial 7: same as 4, except mid discriminator lr = 0.0002 * 0.5

# Trial 6 better (than 4, 5 and 7).

# Trial 8: same as 6, except offspring_mode = "split"
# Trial 9: same as 6, except offspring_mode = "part"

# Will try to increase the loss_rec and loss_cyc a bit.

# Trial 10: hparam_kl 1.0, hparam_rec 2.0, hparam_adv 4.0, hparam_cyc 2.0, offspring_mode == "mean".

# Trial 10 a bit more cleaner than the trial 6, mostly because of the higher rec_loss.

# High network included:


# Trial 1: hparam_kl 1.0, hparam_rec 2.0, hparam_adv 4.0, hparam_cyc 2.0, offspring_mode == "mean", high_disc lr = lr * 0.2
# Trial 2: same as 1 except high_disc lr = lr * 0.5
# Trial 3: same as 1 except high_disc lr = lr * 0.1
# Trial 4: same as 1 except high_disc lr = lr * 0.05
# Trial 5: same, high_disc lr = lr * 0.075, and high gen optimizer only optimize the high gen only, not the mid and low.

# Trial 3_165 is quite good.

trial = 5

hparam_kl = 1.0
hparam_rec = 2.0
hparam_adv = 4.0
hparam_cyc = 2.0

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
z_dim = (1024, 4, 3)
patch_len = (1, 4 ,3)
num_epochs = 200

lr = 0.0002
beta1 = 0.5

f_data = DatasetHDF5("dataset/female_high_1.h5", b_size, True, 100, ['image'])
m_data = DatasetHDF5("dataset/male_high_1.h5", b_size, True, 100, ['image'])
disc_f_data = DatasetHDF5("dataset/female_high_2.h5", b_size, True, 100, ['image'])
disc_m_data = DatasetHDF5("dataset/male_high_2.h5", b_size, True, 100, ['image'])

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

GeneratorFemaleMid = model_mid.Generator().to(device)
GeneratorFemaleMid.apply(weights_init)

GeneratorMaleMid = model_mid.Generator().to(device)
GeneratorMaleMid.apply(weights_init)

DiscriminatorFemaleMid = model_mid.Discriminator().to(device)
DiscriminatorFemaleMid.apply(weights_init)

DiscriminatorMaleMid = model_mid.Discriminator().to(device)
DiscriminatorMaleMid.apply(weights_init)

GeneratorFemaleHigh = model_high.Generator().to(device)
GeneratorFemaleHigh.apply(weights_init)

GeneratorMaleHigh = model_high.Generator().to(device)
GeneratorMaleHigh.apply(weights_init)

DiscriminatorFemaleHigh = model_high.Discriminator().to(device)
DiscriminatorFemaleMid.apply(weights_init)

DiscriminatorMaleHigh = model_high.Discriminator().to(device)
DiscriminatorMaleHigh.apply(weights_init)


loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()

optimizerGenerator = optim.Adam(list(Encoder.parameters()) + list(DecoderFemale.parameters()) + list(DecoderMale.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemale = optim.Adam(DiscriminatorFemale.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorMale = optim.Adam(DiscriminatorMale.parameters(), lr=lr, betas=(beta1, 0.999))

optimizerGeneratorFemaleMid = optim.Adam(list(Encoder.parameters()) + list(DecoderFemale.parameters()) + list(GeneratorFemaleMid.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerGeneratorMaleMid = optim.Adam(list(Encoder.parameters()) + list(DecoderMale.parameters()) + list(GeneratorMaleMid.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemaleMid = optim.Adam(DiscriminatorFemaleMid.parameters(), lr=lr*0.2, betas=(beta1, 0.999))
optimizerDiscriminatorMaleMid = optim.Adam(DiscriminatorMaleMid.parameters(), lr=lr*0.2, betas=(beta1, 0.999))

#optimizerGeneratorFemaleHigh = optim.Adam(list(Encoder.parameters()) + list(DecoderFemale.parameters()) + list(GeneratorFemaleMid.parameters()) + list(GeneratorFemaleHigh.parameters()), lr=lr, betas=(beta1, 0.999))
#optimizerGeneratorMaleHigh = optim.Adam(list(Encoder.parameters()) + list(DecoderMale.parameters()) + list(GeneratorMaleMid.parameters()) + list(GeneratorMaleHigh.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerGeneratorFemaleHigh = optim.Adam(GeneratorFemaleHigh.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerGeneratorMaleHigh = optim.Adam(GeneratorMaleHigh.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemaleHigh = optim.Adam(DiscriminatorFemaleHigh.parameters(), lr=lr*0.075, betas=(beta1, 0.999))
optimizerDiscriminatorMaleHigh = optim.Adam(DiscriminatorMaleHigh.parameters(), lr=lr*0.075, betas=(beta1, 0.999))

iters = 0

print("Starting Training Loop...")
    
log_file = open('log/log_low_mid_high_' + str(trial) + '.txt', 'w')
graph_file = open('graph/graph_low_mid_high_' + str(trial) + '.csv', 'w')
graph_file.write('iters,kl,rec,cyc,real,fake,G,real_mid,fake_mid,G_mid,real_high,fake_high,G_high\n')
epoch = 1
child_index = 0


while True:
    epoch_happened, f_in_img_high = f_data.load_batch('train')
    _, m_in_img_high = m_data.load_batch('train')
    _, disc_f_in_img_high = disc_f_data.load_batch('train')
    _, disc_m_in_img_high = disc_m_data.load_batch('train')

    if f_in_img_high.shape[0] != b_size or m_in_img_high.shape[0] != b_size:
        continue
    if disc_f_in_img_high.shape[0] != b_size or disc_m_in_img_high.shape[0] != b_size:
        continue

    f_in_img_mid = np.empty((b_size, 109, 89, 3), np.float)
    m_in_img_mid = np.empty((b_size, 109, 89, 3), np.float)
    disc_f_in_img_mid = np.empty((b_size, 109, 89, 3), np.float)
    disc_m_in_img_mid = np.empty((b_size, 109, 89, 3), np.float)

    f_in_img = np.empty((b_size, 55, 45, 3), np.float)
    m_in_img = np.empty((b_size, 55, 45, 3), np.float)
    disc_f_in_img = np.empty((b_size, 55, 45, 3), np.float)
    disc_m_in_img = np.empty((b_size, 55, 45, 3), np.float)

    for i in xrange(b_size):
        f_in_img_mid[i] = resize(f_in_img_high[i], (109, 89, 3), preserve_range=True)
        m_in_img_mid[i] = resize(m_in_img_high[i], (109, 89, 3), preserve_range=True)
        disc_f_in_img_mid[i] = resize(disc_f_in_img_high[i], (109, 89, 3), preserve_range=True)
        disc_m_in_img_mid[i] = resize(disc_m_in_img_high[i], (109, 89, 3), preserve_range=True)

        f_in_img[i] = resize(f_in_img_mid[i], (55, 45, 3), preserve_range=True)
        m_in_img[i] = resize(m_in_img_mid[i], (55, 45, 3), preserve_range=True)
        disc_f_in_img[i] = resize(disc_f_in_img_mid[i], (55, 45, 3), preserve_range=True)
        disc_m_in_img[i] = resize(disc_m_in_img_mid[i], (55, 45, 3), preserve_range=True)

        if np.random.random() > 0.5:
            f_in_img[i] = np.fliplr(f_in_img[i])
            f_in_img_mid[i] = np.fliplr(f_in_img_mid[i])
            f_in_img_high[i] = np.fliplr(f_in_img_high[i])
        if np.random.random() > 0.5:
            m_in_img[i] = np.fliplr(m_in_img[i])
            m_in_img_mid[i] = np.fliplr(m_in_img_mid[i])
            m_in_img_high[i] = np.fliplr(m_in_img_high[i])
        if np.random.random() > 0.5:
            disc_f_in_img[i] = np.fliplr(disc_f_in_img[i])
            disc_f_in_img_mid[i] = np.fliplr(disc_f_in_img_mid[i])
            disc_f_in_img_high[i] = np.fliplr(disc_f_in_img_high[i])
        if np.random.random() > 0.5:
            disc_m_in_img[i] = np.fliplr(disc_m_in_img[i])
            disc_m_in_img_mid[i] = np.fliplr(disc_m_in_img_mid[i])
            disc_m_in_img_high[i] = np.fliplr(disc_m_in_img_high[i])

    disc_f_in_img = preprocess_convert_images(disc_f_in_img).to(device)
    disc_m_in_img = preprocess_convert_images(disc_m_in_img).to(device)
    f_in_img = preprocess_convert_images(f_in_img).to(device)
    m_in_img = preprocess_convert_images(m_in_img).to(device)

    disc_f_in_img_mid = preprocess_convert_images(disc_f_in_img_mid).to(device)
    disc_m_in_img_mid = preprocess_convert_images(disc_m_in_img_mid).to(device)
    f_in_img_mid = preprocess_convert_images(f_in_img_mid).to(device)
    m_in_img_mid = preprocess_convert_images(m_in_img_mid).to(device)

    disc_f_in_img_high = preprocess_convert_images(disc_f_in_img_high).to(device)
    disc_m_in_img_high = preprocess_convert_images(disc_m_in_img_high).to(device)
    f_in_img_high = preprocess_convert_images(f_in_img_high).to(device)
    m_in_img_high = preprocess_convert_images(m_in_img_high).to(device)

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

    # Update mid networks.

    with torch.no_grad():
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        offspring_f_out_img = DecoderFemale(z_offspring).detach()
        offspring_m_out_img = DecoderMale(z_offspring).detach()

    # Female Discriminator loss.
    DiscriminatorFemaleMid.zero_grad()
    disc_f_out_mid = DiscriminatorFemaleMid(disc_f_in_img_mid)
    loss_real_D_female_mid = loss_mse(disc_f_out_mid, one_label)
    loss_real_D_female_mid.backward()

    disc_m_out_mid = DiscriminatorFemaleMid(disc_m_in_img_mid)

    with torch.no_grad():
        offspring_f_out_img_mid = GeneratorFemaleMid(offspring_f_out_img).detach()
    disc_offspring_f_out_mid = DiscriminatorFemaleMid(offspring_f_out_img_mid)

    loss_fake_D_female_mid = loss_mse(torch.cat((disc_m_out_mid, disc_offspring_f_out_mid), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_female_mid.backward()

    optimizerDiscriminatorFemaleMid.step()  

    # Male Discriminator loss.
    DiscriminatorMaleMid.zero_grad()
    disc_m_out_mid = DiscriminatorMaleMid(disc_m_in_img_mid)
    loss_real_D_male_mid = loss_mse(disc_m_out_mid, one_label)
    loss_real_D_male_mid.backward()

    disc_f_out_mid = DiscriminatorMaleMid(disc_f_in_img_mid)

    with torch.no_grad():
        offspring_m_out_img_mid = GeneratorMaleMid(offspring_m_out_img).detach()
    disc_offspring_m_out_mid = DiscriminatorMaleMid(offspring_m_out_img_mid)

    loss_fake_D_male_mid = loss_mse(torch.cat((disc_f_out_mid, disc_offspring_m_out_mid), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_male_mid.backward()

    optimizerDiscriminatorMaleMid.step()  

    # Female Offspring Adversarial loss.
    Encoder.zero_grad()
    DecoderFemale.zero_grad()
    GeneratorFemaleMid.zero_grad()

    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
    offspring_f_out_img = DecoderFemale(z_offspring)

    offspring_f_out_img_mid = GeneratorFemaleMid(offspring_f_out_img)
    disc_offspring_f_out_mid = DiscriminatorFemaleMid(offspring_f_out_img_mid)

    loss_G_female_mid = loss_mse(disc_offspring_f_out_mid, one_label)
    loss_G_female_mid.backward()

    optimizerGeneratorFemaleMid.step()

    # Male Offspring Adversarial loss.
    Encoder.zero_grad()
    DecoderMale.zero_grad()
    GeneratorMaleMid.zero_grad()

    z_female = Encoder(f_in_img)
    z_male = Encoder(m_in_img)
    z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
    offspring_m_out_img = DecoderMale(z_offspring)

    offspring_m_out_img_mid = GeneratorMaleMid(offspring_m_out_img)
    disc_offspring_m_out_mid = DiscriminatorMaleMid(offspring_m_out_img_mid)

    loss_G_male_mid = loss_mse(disc_offspring_m_out_mid, one_label)
    loss_G_male_mid.backward()

    optimizerGeneratorMaleMid.step()

    # Update high networks.

    with torch.no_grad():
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        offspring_f_out_img = GeneratorFemaleMid(DecoderFemale(z_offspring)).detach()
        offspring_m_out_img = GeneratorMaleMid(DecoderMale(z_offspring)).detach()

    # Female Discriminator loss.
    DiscriminatorFemaleHigh.zero_grad()
    disc_f_out_high = DiscriminatorFemaleHigh(disc_f_in_img_high)
    loss_real_D_female_high = loss_mse(disc_f_out_high, one_label)
    loss_real_D_female_high.backward()

    disc_m_out_high = DiscriminatorFemaleHigh(disc_m_in_img_high)

    with torch.no_grad():
        offspring_f_out_img_high = GeneratorFemaleHigh(offspring_f_out_img).detach()
    disc_offspring_f_out_high = DiscriminatorFemaleHigh(offspring_f_out_img_high)

    loss_fake_D_female_high = loss_mse(torch.cat((disc_m_out_high, disc_offspring_f_out_high), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_female_high.backward()

    optimizerDiscriminatorFemaleHigh.step()  

    # Male Discriminator loss.
    DiscriminatorMaleHigh.zero_grad()
    disc_m_out_high = DiscriminatorMaleHigh(disc_m_in_img_high)
    loss_real_D_male_high = loss_mse(disc_m_out_high, one_label)
    loss_real_D_male_high.backward()

    disc_f_out_high = DiscriminatorMaleHigh(disc_f_in_img_high)

    with torch.no_grad():
        offspring_m_out_img_high = GeneratorMaleHigh(offspring_m_out_img).detach()
    disc_offspring_m_out_high = DiscriminatorMaleHigh(offspring_m_out_img_high)

    loss_fake_D_male_high = loss_mse(torch.cat((disc_f_out_high, disc_offspring_m_out_high), 0), torch.cat((zero_label, zero_label), 0))
    loss_fake_D_male_high.backward()

    optimizerDiscriminatorMaleHigh.step()  

    # Female Offspring Adversarial loss.
    #Encoder.zero_grad()
    #DecoderFemale.zero_grad()
    #GeneratorFemaleMid.zero_grad()
    GeneratorFemaleHigh.zero_grad()

    with torch.no_grad():
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
        offspring_f_out_img = GeneratorFemaleMid(DecoderFemale(z_offspring)).detach()
        offspring_m_out_img = GeneratorMaleMid(DecoderMale(z_offspring)).detach()

    offspring_f_out_img_high = GeneratorFemaleHigh(offspring_f_out_img)
    disc_offspring_f_out_high = DiscriminatorFemaleHigh(offspring_f_out_img_high)

    loss_G_female_high = loss_mse(disc_offspring_f_out_high, one_label)
    loss_G_female_high.backward()

    optimizerGeneratorFemaleHigh.step()

    # Male Offspring Adversarial loss.
    #Encoder.zero_grad()
    #DecoderMale.zero_grad()
    #GeneratorMaleMid.zero_grad()
    GeneratorMaleHigh.zero_grad()

    #z_female = Encoder(f_in_img)
    #z_male = Encoder(m_in_img)
    #z_offspring = crossover(z_female, z_male, child_index, offspring_mode, device)
    #offspring_m_out_img = GeneratorMaleMid(DecoderMale(z_offspring)).detach()

    offspring_m_out_img_high = GeneratorMaleHigh(offspring_m_out_img)
    disc_offspring_m_out_high = DiscriminatorMaleHigh(offspring_m_out_img_high)

    loss_G_male_high = loss_mse(disc_offspring_m_out_high, one_label)
    loss_G_male_high.backward()

    optimizerGeneratorMaleHigh.step()

       # Output training stats
    if iters % 50 == 0:
        print_str = '[trial: %d][iter: %d][epoch: %d/%d]\nL_KL: %.3f/%.3f\tL_Rec: %.3f/%.3f\tL_Cyc: %.3f/%.3f\nL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f\nmid\tL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f\nhigh\tL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f' % (trial, iters, epoch, num_epochs, loss_kl_female.item(), loss_kl_male.item(), loss_rec_female.item(), loss_rec_male.item(), loss_cyc_female.item(), loss_cyc_male.item(), loss_real_D_female.item(), loss_real_D_male.item(), loss_fake_D_female.item(), loss_fake_D_male.item(), loss_G_female.item(), loss_G_male.item(), loss_real_D_female_mid.item(), loss_real_D_male_mid.item(), loss_fake_D_female_mid.item(), loss_fake_D_male_mid.item(), loss_G_female_mid.item(), loss_G_male_mid.item(), loss_real_D_female_high.item(), loss_real_D_male_high.item(), loss_fake_D_female_high.item(), loss_fake_D_male_high.item(), loss_G_female_high.item(), loss_G_male_high.item())
        print(print_str)
        log_file.write(print_str + '\n')
        log_file.flush()
        graph_str = str(iters) + ',' + str((loss_kl_female.item() + loss_kl_male.item()) * 0.5) + ',' + str((loss_rec_female.item() + loss_rec_male.item()) * 0.5) + ',' + str((loss_cyc_female.item() + loss_cyc_male.item()) * 0.5) + ',' + str((loss_real_D_female.item() + loss_real_D_male.item()) * 0.5) + ',' + str((loss_fake_D_female.item() + loss_fake_D_male.item()) * 0.5) + ',' + str((loss_G_female.item() + loss_G_male.item()) * 0.5) + ',' + str((loss_real_D_female_mid.item() + loss_real_D_male_mid.item()) * 0.5) + ',' + str((loss_fake_D_female_mid.item() + loss_fake_D_male_mid.item()) * 0.5) + ',' + str((loss_G_female_mid.item() + loss_G_male_mid.item()) * 0.5) + ',' + str((loss_real_D_female_high.item() + loss_real_D_male_high.item()) * 0.5) + ',' + str((loss_fake_D_female_high.item() + loss_fake_D_male_high.item()) * 0.5) + ',' + str((loss_G_female_high.item() + loss_G_male_high.item()) * 0.5)
        graph_file.write(graph_str + '\n')
        graph_file.flush()
    
    iters += 1
    child_index = (child_index + 1) % 5
    if epoch_happened == True:
    #if iters > 1 and iters % 50 == 1:
        n_row = 10
        n_col_mid = 10
        n_col_high = 12
        f_in_img_high = postprocess_convert_images(f_in_img_high.cpu())
        m_in_img_high = postprocess_convert_images(m_in_img_high.cpu())
        offspring_female_list_mid = []
        offspring_male_list_mid = []
        offspring_female_list_high = []
        offspring_male_list_high = []
        z_female = Encoder(f_in_img)
        z_male = Encoder(m_in_img)
        for ci in xrange(5):
            z_offspring = crossover(z_female, z_male, ci, offspring_mode, device)
            offspring_f_out_img = GeneratorFemaleMid(DecoderFemale(z_offspring))
            offspring_m_out_img = GeneratorMaleMid(DecoderMale(z_offspring))
            offspring_female_list_mid.append(postprocess_convert_images(offspring_f_out_img.detach().cpu()))
            offspring_male_list_mid.append(postprocess_convert_images(offspring_m_out_img.detach().cpu()))
            offspring_female_list_high.append(postprocess_convert_images(GeneratorFemaleHigh(offspring_f_out_img).detach().cpu()))
            offspring_male_list_high.append(postprocess_convert_images(GeneratorMaleHigh(offspring_m_out_img).detach().cpu()))
        img_row_mid = 109
        img_col_mid = 89
        img_row_high = 218
        img_col_high = 178
        image = np.zeros((img_row_high*n_row, img_col_high*n_col_high+img_col_mid*n_col_mid, 3), dtype=np.uint8)
        for r in xrange(n_row):
            image[r*img_row_high:(r+1)*img_row_high, img_col_high*0:img_col_high*1, :] = f_in_img_high[r]
            image[r*img_row_high:(r+1)*img_row_high, img_col_high*1:img_col_high*2, :] = m_in_img_high[r]
            for ci in xrange(5):
                image[r*img_row_high:r*img_row_high+img_row_mid, img_col_high*2+img_col_mid*ci:img_col_high*2+img_col_mid*(ci+1), :] = offspring_female_list_mid[ci][r]
            for ci in xrange(5):
                image[r*img_row_high:r*img_row_high+img_row_mid, img_col_high*2+img_col_mid*(ci+5):img_col_high*2+img_col_mid*(ci+6), :] = offspring_male_list_mid[ci][r]
            for ci in xrange(5):
                image[r*img_row_high:(r+1)*img_row_high, img_col_high*2+img_col_mid*10+img_col_high*ci:img_col_high*2+img_col_mid*10+img_col_high*(ci+1), :] = offspring_female_list_high[ci][r]
            for ci in xrange(5):
                image[r*img_row_high:(r+1)*img_row_high, img_col_high*2+img_col_mid*10+img_col_high*(ci+5):img_col_high*2+img_col_mid*10+img_col_high*(ci+6), :] = offspring_male_list_high[ci][r]
        imsave('sample_low_mid_high/train_' + str(trial) + "_" + str(epoch) + '.jpg', image)    
        torch.save(Encoder.state_dict(), 'weight_low_mid_high/encoder_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DecoderFemale.state_dict(), 'weight_low_mid_high/decoder_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DecoderMale.state_dict(), 'weight_low_mid_high/decoder_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorFemale.state_dict(), 'weight_low_mid_high/discriminator_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorMale.state_dict(), 'weight_low_mid_high/discriminator_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(GeneratorFemaleMid.state_dict(), 'weight_low_mid_high/generator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(GeneratorMaleMid.state_dict(), 'weight_low_mid_high/generator_mid_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorFemaleMid.state_dict(), 'weight_low_mid_high/discriminator_mid_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorMaleMid.state_dict(), 'weight_low_mid_high/discriminator_mid_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(GeneratorFemaleHigh.state_dict(), 'weight_low_mid_high/generator_high_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(GeneratorMaleHigh.state_dict(), 'weight_low_mid_high/generator_high_m_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorFemaleHigh.state_dict(), 'weight_low_mid_high/discriminator_high_f_' + str(trial) + '_' + str(epoch) + '.pt')
        torch.save(DiscriminatorMaleHigh.state_dict(), 'weight_low_mid_high/discriminator_high_m_' + str(trial) + '_' + str(epoch) + '.pt')
        epoch += 1
        if epoch > num_epochs:
            break

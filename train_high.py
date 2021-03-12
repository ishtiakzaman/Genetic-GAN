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
import model_high as model
from dataset_hdf5 import DatasetHDF5
from utils import preprocess_convert_images, postprocess_convert_images
from utils import crossover

print(torch.__version__)

# Trial 1: hparam_kl 1.0, hparam_rec 2.0, hparam_adv 4.0, hparam_cyc 2.0
# Trial 2: same as 1, except disc_lr = lr * 0.1
# Trial 3: same as 1, except disc_lr = lr * 0.2
# Trial 4: same as 1, except disc_lr = lr * 0.075

# 2 works better than 1, 3, 4.

# Will try old / young version now.

# Trial 5: old / young dataset, hparam_kl 1.0, hparam_rec 2.0, hparam_adv 4.0, hparam_cyc 2.0, disc_lr = lr * 0.1
# Trial 6: old / young dataset, hparam_kl 1.0, hparam_rec 1.5, hparam_adv 4.0, hparam_cyc 1.5, disc_lr = lr * 0.1

# 5 is a bit better than 6.

# Trial 7: same as 5, except change in dynamic switching, for each batch of image, loop through 5 times and backpropagate 5 times.

# 7 nothing special, back to dynamic switching.

# Trial 8: same as 5, with offspring_mode = "split"
# Trial 9: same as 5, with offspring_mode = "part": epoch 91 is good.

# 5, 8, 9 main 3.

# Result:
# 5_101. <= this goes to the paper.
# 8_126.
# 9_91.

# batch size ^^ 32 so far. now will decrease for siamese.

# Trial 10: siamese, 'mean'. hparam_sia = 1.0, b_size = 16, cuda:1

# Trial 191 (dummy): siamese, 'mean'. hparam_sia = 1.0, b_size = 16, cuda:6
# Trial 192 (dummy): siamese, 'mean'. hparam_sia = 1.0, b_size = 16, cuda:7

trial = 10

hparam_kl = 1.0
hparam_rec = 2.0 
hparam_adv = 4.0
hparam_cyc = 2.0
hparam_sia = 1.0

device = torch.device("cuda:6")

#offspring_mode = "part"
#offspring_mode = "split"
offspring_mode = "mean"

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 16
z_dim = (1024, 4, 3)
patch_len = (1, 4 ,3)
num_epochs = 1000

lr = 0.0002
beta1 = 0.5

f_data = DatasetHDF5("dataset/female_young_old.h5", b_size, True, 100, ['image'])
m_data = DatasetHDF5("dataset/male_young_old.h5", b_size, True, 100, ['image'])
disc_f_data = DatasetHDF5("dataset/female_young.h5", b_size, True, 100, ['image'])
disc_m_data = DatasetHDF5("dataset/male_young.h5", b_size, True, 100, ['image'])

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

sys.path.insert(0, 'siamese')
import siamese_model
Siamese = siamese_model.Siamese().to(device)
Siamese.load_state_dict(torch.load('siamese/weight/s_1_200.pt', map_location='cpu'))

loss_bce = nn.BCELoss()
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()

optimizerGenerator = optim.Adam(list(Encoder.parameters()) + list(DecoderFemale.parameters()) + list(DecoderMale.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerDiscriminatorFemale = optim.Adam(DiscriminatorFemale.parameters(), lr=lr*0.1, betas=(beta1, 0.999))
optimizerDiscriminatorMale = optim.Adam(DiscriminatorMale.parameters(), lr=lr*0.1, betas=(beta1, 0.999))

iters = 0

print("Starting Training Loop...")
    
log_file = open('log/log_high_' + str(trial) + '.txt', 'w')
graph_file = open('graph/graph_high_' + str(trial) + '.csv', 'w')
graph_file.write('iters,total,kl,rec,cyc,real,fake,G,siamese\n')
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

    # Female KL, Reconstruction, Cycle, and Offspring Adversarial loss.
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

    with torch.no_grad():
        ef, eo = Siamese(f_in_img, offspring_f_out_img)
        ef = ef.detach().cpu()
        eo = eo.detach().cpu()
        em, _ = Siamese(m_in_img, offspring_f_out_img)
        em = em.detach().cpu()
    loss_siamese_female = loss_mse(ef, eo) * (1.0 - 0.25 * child_index) + loss_mse(em, eo) * (0.25 * child_index)

    loss_total_female = hparam_rec * loss_rec_female + hparam_kl * loss_kl_female + hparam_adv * loss_G_female + hparam_cyc * loss_cyc_female + hparam_sia * loss_siamese_female
    loss_total_female.backward()

    optimizerGenerator.step()

    # Male KL, Reconstruction, Cycle, and Offspring Adversarial loss.
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

    with torch.no_grad():
        ef, eo = Siamese(f_in_img, offspring_m_out_img)
        ef = ef.detach().cpu()
        eo = eo.detach().cpu()
        em, _ = Siamese(m_in_img, offspring_m_out_img)
        em = em.detach().cpu()
    loss_siamese_male = loss_mse(ef, eo) * (1.0 - 0.25 * child_index) + loss_mse(em, eo) * (0.25 * child_index)

    loss_total_male = hparam_rec * loss_rec_male + hparam_kl * loss_kl_male + hparam_adv * loss_G_male + hparam_cyc * loss_cyc_male + hparam_sia * loss_siamese_male
    loss_total_male.backward()

    optimizerGenerator.step()

    loss_total = loss_total_female + loss_total_male
          
    # Output training stats
    if iters % 50 == 0:
        print_str = '[trial: %d][iter: %d][epoch: %d/%d]\tL_Total: %.3f\nL_KL: %.3f/%.3f\tL_Rec: %.3f/%.3f\tL_Cyc: %.3f/%.3f\nL_Real_D: %.3f/%.3f\tL_Fake_D: %.3f/%.3f\tL_Adv_G: %.3f/%.3f\tL_Sia: %.3f/%.3f' % (trial, iters, epoch, num_epochs, loss_total.item(), loss_kl_female.item(), loss_kl_male.item(), loss_rec_female.item(), loss_rec_male.item(), loss_cyc_female.item(), loss_cyc_male.item(), loss_real_D_female.item(), loss_real_D_male.item(), loss_fake_D_female.item(), loss_fake_D_male.item(), loss_G_female.item(), loss_G_male.item(), loss_siamese_female, loss_siamese_male)
        print(print_str)
        log_file.write(print_str + '\n')
        log_file.flush()
        graph_str = str(iters) + ',' + str(loss_total.item()) + ',' + str((loss_kl_female.item() + loss_kl_male.item()) * 0.5) + ',' + str((loss_rec_female.item() + loss_rec_male.item()) * 0.5) + ',' + str((loss_cyc_female.item() + loss_cyc_male.item()) * 0.5) + ',' + str((loss_real_D_female.item() + loss_real_D_male.item()) * 0.5) + ',' + str((loss_fake_D_female.item() + loss_fake_D_male.item()) * 0.5) + ',' + str((loss_G_female.item() + loss_G_male.item()) * 0.5) + ',' + str((loss_siamese_female.item() + loss_siamese_male.item()) * 0.5)
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
        img_row = 218
        img_col = 178
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
        imsave('sample_high/train_' + str(trial) + "_" + str(epoch) + '.jpg', image)    
        #if epoch % 5 == 0:
        torch.save(Encoder.state_dict(), 'weight_high/' + str(trial) + '_encoder_' + str(epoch) + '.pt')
        torch.save(DecoderFemale.state_dict(), 'weight_high/' + str(trial) + '_decoder_f_' + str(epoch) + '.pt')
        torch.save(DecoderMale.state_dict(), 'weight_high/' + str(trial) + '_decoder_m_' + str(epoch) + '.pt')
        torch.save(DiscriminatorFemale.state_dict(), 'weight_high/' + str(trial) + '_discriminator_f_' + str(epoch) + '.pt')
        torch.save(DiscriminatorMale.state_dict(), 'weight_high/' + str(trial) + '_discriminator_m_' + str(epoch) + '.pt')
        epoch += 1
        if epoch > num_epochs:
            break

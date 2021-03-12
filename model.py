import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 0, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 0, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 0, bias=False)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 0, bias=False)

        self.n_res_blk = 4
        self.res_conv = nn.Conv2d(1024, 1024, 3, 1, 0, bias=False)

        self.ref_pad1 = nn.ReflectionPad2d(1)
        self.ref_pad2 = nn.ReflectionPad2d(2)
        self.ref_pad3 = nn.ReflectionPad2d(3)

        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_1024 = nn.BatchNorm2d(1024)

    def residual_block_with_activation(self, ip):
        # input is 1024 x 4 x 3
        x = F.leaky_relu(self.bn_1024(self.res_conv(self.ref_pad1(ip))), 0.2)
        x = F.leaky_relu(self.bn_1024(self.res_conv(self.ref_pad1(x))), 0.2)
        # output is 1024 x 4 x 3
        return torch.add(ip, x)
        
    def residual_block_without_activation(self, ip):
        # input is 1024 x 4 x 3
        x = F.leaky_relu(self.bn_1024(self.res_conv(self.ref_pad1(ip))), 0.2)
        x = self.bn_1024(self.res_conv(self.ref_pad1(x)))
        # output is 1024 x 4 x 3
        return torch.add(ip, x)

    def encoder(self, x):
        # input is 3 x 55 x 45
        x = F.leaky_relu(self.bn_64(self.conv1(self.ref_pad3(x))), 0.2)
        # state size 64 x 55 x 45
        x = F.leaky_relu(self.bn_128(self.conv2(self.ref_pad1(x))), 0.2)
        # state size 128 x 28 x 23
        x = F.leaky_relu(self.bn_256(self.conv3(self.ref_pad1(x))), 0.2)
        # state size 256 x 14 x 12
        x = F.leaky_relu(self.bn_512(self.conv4(self.ref_pad1(x))), 0.2)
        # state size 512 x 7 x 6
        x = F.leaky_relu(self.bn_1024(self.conv5(self.ref_pad1(x))), 0.2)
        # state size 1024 x 4 x 3

        # Series of 4 residual blocks.
        for _ in xrange(self.n_res_blk):
            x = self.residual_block_with_activation(x)
            # state size 1024 x 4 x 3

        # Last residual block to produce the z.
        x = self.residual_block_without_activation(x)
        # state size 1024 x 4 x 3
        return x

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.conv_trans2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.conv_trans3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.conv_trans4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)

        self.conv = nn.Conv2d(64, 3, 7, 1, 0, bias=False)

        self.n_res_blk = 4
        self.res_conv = nn.Conv2d(1024, 1024, 3, 1, 0, bias=False)

        self.ref_pad1 = nn.ReflectionPad2d(1)
        self.ref_pad2 = nn.ReflectionPad2d(2)
        self.ref_pad3 = nn.ReflectionPad2d(3)

        self.bn_3 = nn.BatchNorm2d(3)
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_1024 = nn.BatchNorm2d(1024)

    def residual_block_with_activation(self, ip):
        # input is 1024 x 4 x 3
        x = F.leaky_relu(self.bn_1024(self.res_conv(self.ref_pad1(ip))), 0.2)
        x = F.leaky_relu(self.bn_1024(self.res_conv(self.ref_pad1(x))), 0.2)
        # output is 1024 x 4 x 3
        return torch.add(ip, x)

    def decoder(self, x):
        # input is z, 1024 x 4 x 3 latent vector.

        # Series of 4 resisual blocks.
        for _ in xrange(self.n_res_blk):
            x = self.residual_block_with_activation(x)
            # state size 1024 x 4 x 3

        x = F.leaky_relu(self.bn_512(self.conv_trans1(x)), 0.2)
        # state size 512 x 8 x 6
        x = x[:,:,0:7,:]
        # state size 512 x 7 x 6
        x = F.leaky_relu(self.bn_256(self.conv_trans2(x)), 0.2)
        # state size 256 x 14 x 12
        x = F.leaky_relu(self.bn_128(self.conv_trans3(x)), 0.2)
        # state size 128 x 28 x 24
        x = x[:,:,:,0:23]
        # state size 128 x 28 x 23
        x = F.leaky_relu(self.bn_64(self.conv_trans4(x)), 0.2)
        # state size 64 x 56 x 46
        x = x[:,:,0:55,0:45]
        # state size 64 x 55 x 45
        x = torch.tanh(self.bn_3(self.conv(self.ref_pad3(x))))
        # state size 3 x 55 x 45
        return x

    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, 4, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(256, 1, 4, 1, 1, bias=False)

        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_1024 = nn.BatchNorm2d(1024)

    def forward(self, x):
        # input is 3 x 55 x 45
        x = F.leaky_relu(self.conv1(x), 0.2)
        # state size. 64 x 27 x 22
        x = F.leaky_relu(self.bn_128(self.conv2(x)), 0.2)
        # state size. 128 x 13 x 11
        x = F.leaky_relu(self.bn_256(self.conv3(x)), 0.2)
        # state size. 256 x 6 x 5
        x = F.leaky_relu(self.bn_256(self.conv4(x)), 0.2)
        # state size. 256 x 5 x 4
        x = torch.sigmoid(self.conv5(x))
        # state size. 1 x 4 x 3
        return x

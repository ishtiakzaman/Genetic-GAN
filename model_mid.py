import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 0, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 0, bias=False)

        self.conv_trans1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.conv_trans2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.conv_trans3 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)

        self.conv = nn.Conv2d(32, 3, 7, 1, 0, bias=False)

        self.n_res_blk = 5
        self.res_conv = nn.Conv2d(256, 256, 3, 1, 0, bias=False)

        self.ref_pad1 = nn.ReflectionPad2d(1)
        self.ref_pad2 = nn.ReflectionPad2d(2)
        self.ref_pad3 = nn.ReflectionPad2d(3)

        self.bn_3 = nn.BatchNorm2d(3)
        self.bn_32 = nn.BatchNorm2d(32)
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_1024 = nn.BatchNorm2d(1024)

    def residual_block(self, ip):
        # input is 256 x 14 x 12
        x = F.leaky_relu(self.bn_256(self.res_conv(self.ref_pad1(ip))), 0.2)
        x = F.leaky_relu(self.bn_256(self.res_conv(self.ref_pad1(x))), 0.2)
        # output is 256 x 14 x 12
        return torch.add(ip, x)
        
    def forward(self, x):
        # input is 3 x 55 x 45
        x = F.leaky_relu(self.bn_64(self.conv1(self.ref_pad3(x))), 0.2)
        # state size 64 x 55 x 45
        x = F.leaky_relu(self.bn_128(self.conv2(self.ref_pad1(x))), 0.2)
        # state size 128 x 28 x 23
        x = F.leaky_relu(self.bn_256(self.conv3(self.ref_pad1(x))), 0.2)
        # state size 256 x 14 x 12

        # Series of 5 residual blocks.
        for _ in xrange(self.n_res_blk):
            x = self.residual_block(x)
            # state size 256 x 14 x 12

        x = F.leaky_relu(self.bn_128(self.conv_trans1(x)), 0.2)
        # state size 128 x 28 x 24
        x = x[:,:,:,0:23]
        # state size 128 x 28 x 23
        x = F.leaky_relu(self.bn_64(self.conv_trans2(x)), 0.2)
        # state size 64 x 56 x 46
        x = x[:,:,0:55,0:45]
        # state size 64 x 55 x 45
        x = F.leaky_relu(self.bn_32(self.conv_trans3(x)), 0.2)
        # state size 32 x 110 x 90
        x = x[:,:,0:109,0:89]
        # state size 32 x 109 x 89
        x = torch.tanh(self.bn_3(self.conv(self.ref_pad3(x))))
        # state size 3 x 109 x 89

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, 4, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(512, 1, 4, 1, 1, bias=False)

        self.bn_128 = nn.BatchNorm2d(128)
        self.bn_256 = nn.BatchNorm2d(256)
        self.bn_512 = nn.BatchNorm2d(512)

    def forward(self, x):
        # input is 3 x 109 x 89
        x = F.leaky_relu(self.conv1(x), 0.2)
        # state size. 64 x 54 x 44
        x = F.leaky_relu(self.bn_128(self.conv2(x)), 0.2)
        # state size. 128 x 27 x 22
        x = F.leaky_relu(self.bn_256(self.conv3(x)), 0.2)
        # state size. 256 x 13 x 11
        x = F.leaky_relu(self.bn_512(self.conv4(x)), 0.2)
        # state size. 512 x 6 x 5
        x = F.leaky_relu(self.bn_512(self.conv5(x)), 0.2)
        # state size. 512 x 5 x 4
        x = torch.sigmoid(self.conv6(x))
        # state size. 1 x 4 x 3
        return x

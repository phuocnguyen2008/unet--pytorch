"""
    Define model architecture using paper U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.downsampling = nn.MaxPool2d(kernel_size=2)

        self.down_block1 = DoubleConv(1, 64)
        self.down_block2 = DoubleConv(64, 128)
        self.down_block3 = DoubleConv(128, 256)
        self.down_block4 = DoubleConv(256, 512)
        self.down_block5 = DoubleConv(512, 512)

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')

        self.up_block1 = DoubleConv(1024, 512)
        self.up_block2 = DoubleConv(512, 256)
        self.up_block3 = DoubleConv(256, 128)
        self.up_block4 = DoubleConv(128, 64)

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        
        out1 = self.down_block1(x) # 64 x 568 x 568
        out2 = self.down_block2(self.downsampling(out1)) # 128 x 280 x 280
        out3 = self.down_block3(self.downsampling(out2)) # 256 x 136 x 136
        out4 = self.down_block4(self.downsampling(out3)) # 512 x 64 x 64
        out5 = self.down_block5(self.downsampling(out4)) # 512 x 28 x 28

        out6_ = self.upsampling(out5) # 512 x 56 x 56
        out4_ = out4[:, :, 4:60, 4:60] # 512 x 56 x 56
        out6 = self.up_block1(torch.cat((out4_, out6_), dim=1))

        out7_ = self.upsampling(out6)
        out3_ = out3[:, :, 16:120, 16:120]
        out7 = self.up_block2(torch.cat((out3_, out7_), dim=1))

        out8_ = self.upsampling(out7)
        out2_ = out2[:, :, 40:240, 40:240]
        out8 = self.up_block3(torch.cat((out2_, out8_), dim=1))

        out9_ = self.upsampling(out8)
        out1_ = out1[:, :, 88:480, 88:480]
        out9 = self.up_block4(torch.cat((out1_, out9_), dim=1)) # 64 x 388 x 388

        out = self.conv11(out9)

        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
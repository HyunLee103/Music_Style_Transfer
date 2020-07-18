import os
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from tqdm import tqdm

# encoder
self.n_blocks = args.encoder_blocks # 3
self.n_layers = args.encoder_layers # 10
self.channels = args.encoder_channels # 128
self.latent_channels = args.latent_d # 64
self.activation = args.encoder_func # relu

# decoder
self.blocks = args.blocks  # 4
self.layer_num = args.layers  # 14
self.kernel_size = args.kernel_size # 2
self.skip_channels = args.skip_channels # 128
self.residual_channels = args.residual_channels # 128
self.cond_channels = args.latent_d # 64



# encoder
class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, activation='relu', padding=1, kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels

        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'glu':
            self.activation = torch.glu
            in_channels = channels // 2

        self.left_pad = left_pad
        self.dilated_conv = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=1,
                                      padding=dilation * padding, dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels,
                                  kernel_size=1, bias=True)

    def forward(self, input):
        x = input

        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        x = self.dilated_conv(x)
        x = self.activation(x)
        x = self.conv_1x1(x)

        return input + x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_blocks = args.encoder_blocks
        self.n_layers = args.encoder_layers
        self.channels = args.encoder_channels
        self.latent_channels = args.latent_d
        self.activation = args.encoder_func

        try:
            self.encoder_pool = args.encoder_pool
        except AttributeError:
            self.encoder_pool = 800

        layers = []
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                layers.append(DilatedResConv(self.channels, dilation, self.activation))
        self.dilated_convs = nn.Sequential(*layers)

        self.start = nn.Conv1d(1, self.channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv_1x1 = nn.Conv1d(self.channels, self.latent_channels, 1)
        self.pool = nn.AvgPool1d(self.encoder_pool)

    def forward(self, x):
        x = x / 255 - .5
        if x.dim() < 3:
            x = x.unsqueeze(1)

        x = self.start(x)
        x = self.dilated_convs(x)
        x = self.conv_1x1(x)
        x = self.pool(x)

        return x
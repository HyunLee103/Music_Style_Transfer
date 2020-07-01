import os
import numpy as np
from Spectral_norm import *
import torch
import torch.nn as nn
from utils import *
from tensordot import *

class DECBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="inorm", relu=0.0,output_padding=0):
        super().__init__()

        layers = []
        ## nn.layer에 SpectralNorm 적용
        layers += [SpectralNorm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding, output_padding = output_padding,
                             bias=bias))]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="inorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias))]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class DenseSN(nn.Linear):
    def __init__(self, input_shape):
        super(DenseSN, self).__init__(in_features=input_shape, out_features=1)
        
        self.u = torch.nn.Parameter(data=torch.zeros((1,self.weight.shape[-1]))
                                         ,requires_grad=False)
        torch.nn.init.normal_(self.u.data, mean=0.5, std=0.5)
        
        self.u.data.uniform_(0, 1)
    
    def compute_spectral_norm(self, W, new_u, W_shape):
        new_v = F.normalize(torch.matmul(new_u, torch.transpose(W,0,1)), p=2)
        new_u = F.normalize(torch.matmul(new_v, W), p=2)
            
        sigma = torch.matmul(W, torch.transpose(new_u,0,1))
        W_bar = W/sigma

        self.u = torch.nn.Parameter(data=new_u)
        W_bar = W_bar.reshape(W_shape)

        return W_bar
    
    def forward(self, inputs):
        W_shape = self.weight.shape
        W_reshaped = self.weight.reshape((-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        
        rank = len(inputs.shape)
        
        if rank > 2:
            #Thanks to deanmark on GitHub for pytorch tensordot function
            outputs = tensordot_pytorch(inputs, new_kernel, [[rank-1],[0]])
        else:
            outputs = torch.matmul(inputs, torch.transpose(new_kernel,0,1))
            
        outputs = torch.tanh(outputs)
        
        return outputs
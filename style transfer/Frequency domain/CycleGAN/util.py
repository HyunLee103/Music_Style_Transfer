import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

import os
import torch.nn as nn

## 네트워크 저장하기
## cycleGAN은 네트워크가 4개니까 save 함수도 각 네트워크를 저장하게 변경
def save(ckpt_dir, netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG_A2B': netG_A2B.state_dict(), 'netG_B2A': netG_B2A.state_dict(),
                'netD_B': netD_B.state_dict(), 'netD_B': netD_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(), 'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## 네트워크 불러오기
## 저장된 네트워크 불러오는것도 마찬가지로 4개의 네트워크 각각 불러옴
def load(ckpt_dir, netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B):
    if not os.path.exists(ckpt_dir):
        epoch = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG_A2B.load_state_dict(dict_model['netG_A2B'])
    netG_B2A.load_state_dict(dict_model['netG_B2A'])
    netD_A.load_state_dict(dict_model['netD_A'])
    netD_B.load_state_dict(dict_model['netD_B'])
    optimizer_G.load_state_dict(dict_model['optimizer_G'])
    optimizer_D_A.load_state_dict(dict_model['optimizer_D_A'])
    optimizer_D_B.load_state_dict(dict_model['optimizer_D_B'])

    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B, epoch


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)
       

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
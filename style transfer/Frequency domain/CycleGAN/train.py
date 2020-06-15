import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from dataset import Dataset
from model import *
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--ckpt_dir',type=str,default = '/content/drive/My Drive/ADV_Project_Music_style_transform/timbreTron_cycleGAN/ckpt')

opt = parser.parse_args()
print(opt)

data_dir = opt.dataroot
ckpt_dir = opt.ckpt_dir
result_dir_test = '/content/drive/My Drive/ADV_Project_Music_style_transform/timbreTron_cycleGAN/result'
batch_size = opt.batchSize
device = torch.device('cuda') # if torch.cuda.is_available() else 'cpu')

# helper
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
cmap = None

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
netD_A = Discriminator(opt.input_nc,opt.input_nc).to(device)
netD_B = Discriminator(opt.output_nc,opt.input_nc).to(device)
# A domain이 label인 discriminator
# B domain이 label인 discriminator

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.BCELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_identity = torch.nn.L1Loss().to(device)

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0, 0.9))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0, 0.9))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0, 0.9))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transform_ = [transforms.ToTensor()]
dataloader = DataLoader(Dataset(data_dir, transform=transform_),batch_size = opt.batchSize,shuffle=True)



###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):

    for batch, data in enumerate(dataloader):
        # Set model input
        real_A = data['pop'].to(device)
        real_B = data['jazz'].to(device)

        real_A = real_A.float().cuda()
        real_B = real_B.float().cuda()

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_fake))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_fake))
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

    print("TRAIN: EPOCH %04d | G_LOSS %.4f  | D_fake_LOSS %.4f | D_real_LOSS %.4f " %(epoch, loss_G,loss_D_fake ,loss_D_real))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save output
    if epoch % 10 == 0:
        genre_1 = fn_tonumpy(real_A)
        genre_2 = fn_tonumpy(real_B)
        output = fn_tonumpy(fn_denorm(fake_B,mean=0.5,std=0.5)) # generator로 생성된 output은 마지막 layer에 tanh를 거치며 normalize된다.

        for j in range(genre_2.shape[0]):
            # id = batch_size * (batch - 1) + j

            label_ = genre_1[j]
            input_ = genre_2[j]
            output_ = output[j]

            np.save(os.path.join(result_dir_test, 'numpy', '%04d_pop.npy' % epoch), label_)
            np.save(os.path.join(result_dir_test, 'numpy', '%04d_jazz.npy' % epoch), input_)
            np.save(os.path.join(result_dir_test, 'numpy', '%04d_pop2jazz.npy' % epoch), output_)

    # Save models checkpoints
    if epoch % 2 == 0:
        save(ckpt_dir=ckpt_dir, netG_A2B=netG_A2B, netG_B2A=netG_B2A, netD_A=netD_A, netD_B=netD_B, optimizer_G=optimizer_G, optimizer_D_A=optimizer_D_A, optimizer_D_B=optimizer_D_B, epoch=epoch)


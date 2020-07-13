# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import itertools
import os
import IPython

def play_music(mus, sr=32000):
    return IPython.display.display(IPython.display.Audio(mus, rate=sr))

"""
# mu-law function in 'music-translation' git
# Is it same with librosa.core.mu_compress()?
def mu_law(x, mu=255):
    x = numpy.clip(x, -1, 1)
    x_mu = numpy.sign(x) * numpy.log(1 + mu*numpy.abs(x))/numpy.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = numpy.array(x).astype(numpy.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return numpy.sign(y) * (1./mu) * ((1. + mu)**numpy.abs(y) - 1.)
"""

"""
--CausalConv1d
--WavenetLayer  
--WaveNet
remove condition func. We don't use condition
"""

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 dilation=1,
                 **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
            **kwargs)

    def forward(self, input):
        out = super(CausalConv1d, self).forward(input)
        return out[:, :, :-self.padding[0]]


class WavenetLayer(nn.Module):
    def __init__(self, residual_channels, skip_channels,
                 kernel_size=2, dilation=1):
        super(WavenetLayer, self).__init__()

        self.causal = CausalConv1d(residual_channels, residual_channels,
                                   kernel_size, dilation=dilation, bias=True)
        # self.condition = nn.Conv1d(cond_channels, 2 * residual_channels,
        #                            kernel_size=1, bias=True)
        self.residual = nn.Conv1d(residual_channels, residual_channels,
                                  kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels,
                              kernel_size=1, bias=True)

    # def _condition(self, x, c, f):
    #     c = f(c)
    #     x = x + c
    #     return x

    def forward(self, x):
        x = self.causal(x)
        # if c is not None:
        #     x = self._condition(x, c, self.condition)

        assert x.size(1) % 2 == 0
        gate, output = x.chunk(2, 1)
        gate = torch.sigmoid(gate)
        output = torch.tanh(output)
        x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class WaveNet(nn.Module):
    def __init__(self, blocks, channels, layer_num, shift_input=False, generator=True, out_func='tanh'):
        super().__init__()

        self.blocks = blocks
        self.layer_num = layer_num
        self.kernel_size = 2
        self.channels = channels
        self.skip_channels = channels
        self.residual_channels = channels
        # self.cond_channels = args.latent_d
        self.classes = 1
        self.shift_input = shift_input
        self.out_func = out_func
        self.generator = generator


        self.first_conv = CausalConv1d(1, self.channels, kernel_size=self.kernel_size)

        layers = []
        for _ in range(self.blocks):
            for i in range(self.layer_num):
                dilation = 2 ** i
                layers.append(CausalConv1d(self.channels, self.channels, self.kernel_size, dilation))
        self.layers = nn.ModuleList(layers)

        # self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        # self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)
        # self.fc = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.channels, self.classes, kernel_size=1)

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        if (not 'Half' in x.type()) and (not 'Float' in x.type()):
            x = x.float()

        # x = x / 255 - 0.5

        if self.shift_input:
            x = self.shift_right(x)

        # if c is not None:
        #     c = self._upsample_cond(x, c)

        residual = F.relu(self.first_conv(x))
        #skip = self.skip_conv(residual)

        # for layer in self.layers:
        #     r, s = layer(residual)
        #     residual = residual + r
        #     skip = skip + s
    
        for layer in self.layers:
            residual = F.relu(layer(residual))

        # skip = F.relu(skip)
        # skip = self.fc(skip)
        # if c is not None:
        #     skip = self._condition(skip, c, self.condition)
        # skip = F.relu(skip)
        # skip = self.logits(skip)

        skip = self.logits(residual)
        
        if not self.generator:
            return torch.sigmoid(skip.mean(2))

        if self.out_func == 'tanh':
            return torch.tanh(skip)
        
        return skip
    
    @staticmethod
    def shift_right(x):
        x = F.pad(x, (1, 0))
        return x[:, :, :-1]

    # def _condition(self, x, c, f):
    #     c = f(c)
    #     x = x + c
    #     return x

    # @staticmethod
    # def _upsample_cond(x, c):
    #     bsz, channels, length = x.size()
    #     cond_bsz, cond_channels, cond_length = c.size()
    #     assert bsz == cond_bsz

    #     if c.size(2) != 1:
    #         c = c.unsqueeze(3).repeat(1, 1, 1, length // cond_length)
    #         c = c.view(bsz, cond_channels, length)

    #     return c


"""
--ZDiscriminator--
args.d_channels == 100
args.latent_d == 128
"""
class ZDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_classes = 2

        convs = []
        for i in range(args.d_layers):  # d_layesrs default 3
            in_channels = args.latent_d if i == 0 else args.d_channels
            convs.append(nn.Conv1d(in_channels, args.d_channels, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(args.d_channels, self.n_classes, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=args.p_dropout_discriminator)

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)

        mean = logits.mean(2)
        return mean

class WaveData(Dataset):
    def __init__(self, jazz_path, rock_path):
        super().__init__()
        print("Data load...")
        self.jazz = np.load(jazz_path)[:, :128000]
        self.rock = np.load(rock_path)[:, :128000]
        print("done")


    def __len__(self):
        return min(len(self.jazz), len(self.rock))
    
    def __getitem__(self, index):
        return {'jazz': self.jazz[index], 'rock': self.rock[index]}

# params
model_path = "/content/drive/My Drive/models"
train_keyword = "lr_1e-4_tanh_batch8"
start_epoch = 1
num_epoch = 1000
batch_size = 8
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(
f"""
Train keyword {train_keyword}
Batch size {batch_size}
Learning rate {lr}
Device {device}
""")

# mkdir
save_path = os.path.join(model_path, train_keyword)

if os.path.isdir(save_path) and os.listdir(save_path) != []:
    RuntimeError("Same train_keyword.")
else:
    os.mkdir(save_path)

# load data
data = WaveData('/content/drive/My Drive/jazz_all_gtzan_16000.npy', '/content/drive/My Drive/rock_all_gtzan_16000.npy')

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

for d in dataloader:
    break

d['jazz'].shape



arr = np.array(d['rock'])

play_music(arr[4], sr=16000)

max(arr[0])

# load models
print("Model to device")
G_A2B = WaveNet(blocks=2, channels=32, layer_num=3, generator=True, out_func='tanh').to(device)
G_B2A = WaveNet(blocks=2, channels=32, layer_num=3, generator=True, out_func='tanh').to(device)
D_A = WaveNet(blocks=1, channels=32, layer_num=4, generator=False).to(device)
D_B = WaveNet(blocks=1, channels=32, layer_num=4, generator=False).to(device)
print("done")

criterion_identity = torch.nn.L1Loss().to(device)
criterion_GAN = torch.nn.BCELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)

optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

print("Training")
batch_len = len(dataloader)
for epoch in range(start_epoch, num_epoch + 1):
    print(f"Epoch {epoch}")
    
    for i, data in enumerate(dataloader):
        # Set model input
        real_A = data['jazz'].unsqueeze(1).to(device)
        real_B = data['rock'].unsqueeze(1).to(device)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # # Identity loss
        # # G_A2B(B) should equal B if real B is fed
        # same_B = G_A2B(real_B)
        # loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # # G_B2A(A) should equal A if real A is fed
        # same_A = G_B2A(real_A)
        # loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = G_A2B(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        fake_A = G_B2A(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

        # Cycle loss
        recovered_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_fake))

        # Fake loss
        pred_fake = D_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = D_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_fake))
        
        # Fake loss
        pred_fake = D_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        ###################################

        print(f"Epoch: {epoch} -- Batch {i+1}/{batch_len}")

        print(f'G_Loss : {loss_G.item():0.4}')
        print(f'D_loss : {loss_D_A.item() + loss_D_B.item():0.4}\n')

    if not epoch % 100:
        torch.save({
            'epoch': epoch,
            'G_A2B': G_A2B.state_dict(),
            'G_B2A': G_B2A.state_dict(),
            'D_A': D_A.state_dict(),
            'D_B': D_B.state_dict(),
            'optim_D_A': optimizer_D_A.state_dict(),
            'optim_D_B': optimizer_D_B.state_dict(),
            'optim_G': optimizer_G.state_dict()
        }, os.path.join(save_path, f"{epoch}.pth"))
    
    if not epoch % 50:
        tmp = fake_A.detach().cpu().numpy()[0][0]
        librosa.output.write_wav(f"/content/drive/My Drive/samples/fake_A_epoch{epoch}.wav", tmp, 16000)


del G_A2B, G_B2A, D_A, D_B, optimizer_D_B, optimizer_D_A, optimizer_G, criterion_cycle, criterion_GAN, criterion_identity
torch.cuda.empty_cache()
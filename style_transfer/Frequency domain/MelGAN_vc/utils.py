#Imports

from __future__ import print_function, division
from glob import glob
import scipy
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
from PIL import Image
import imageio
import librosa
# import librosa.display
from librosa.feature import melspectrogram
import os
import time
import IPython
from torch.autograd import Variable

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#from tensordot_pytorch import tensordot_pytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparameters

hop=192               #hop size (window size = 6*hop)
sr=22050              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

shape=96             #length of time axis of split specrograms to feed to generator            
vec_len=128           #length of vector generated by siamese vector
bs = 64               #batch size
delta = 2.            #constant for siamese loss

#There seems to be a problem with Tensorflow STFT, so we'll be using pytorch to handle offline mel-spectrogram generation and waveform reconstruction
#For waveform reconstruction, a gradient-based method is used:

''' Decorsière, Rémi, Peter L. Søndergaard, Ewen N. MacDonald, and Torsten Dau. 
"Inversion of auditory spectrograms, traditional spectrograms, and other envelope representations." 
IEEE/ACM Transactions on Audio, Speech, and Language Processing 23, no. 1 (2014): 46-56.'''

#ORIGINAL CODE FROM https://github.com/yoyololicon/spectrogram-inversion

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import heapq
import torchaudio
from torchaudio.transforms import MelScale, Spectrogram

#uncomment if you have a gpu
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

specobj = Spectrogram(n_fft=6*hop, win_length=6*hop, hop_length=hop, pad=0, power=2, normalized=True)
specfunc = specobj.forward
melobj = MelScale(n_mels=hop, sample_rate=sr, f_min=0.)
melfunc = melobj.forward

def melspecfunc(waveform):
    specgram = specfunc(waveform)
    mel_specgram = melfunc(specgram)
    return mel_specgram

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=100, tol=1e-6, verbose=1, evaiter=10, lr=0.003):

    spec = torch.Tensor(spec).to('cpu')
    samples = (spec.shape[-1]*hop)-hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    for i in range(maxiter):
        print(i)
        optimizer.zero_grad()
        V = transform_fn(x)
        loss = criterion(V, T)
        loss.backward()
        optimizer.step()
        lr = lr*0.9999
        for param_group in optimizer.param_groups:
            optimizer.lr = lr

        if i % evaiter == evaiter - 1:
            with torch.no_grad():
                V = transform_fn(x)
                l2_loss = criterion(V, spec).item()

    return x.detach().view(-1).cpu()

def normalize(S):
    return np.clip((((S - min_level_db) / -min_level_db)*2.)-1., -1, 1)

def denormalize(S):
    return (((np.clip(S, -1, 1)+1.)/2.) * -min_level_db) + min_level_db

def prep(wv,hop=192):
    S = np.array(torch.squeeze(melspecfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
    S = librosa.power_to_db(S)-ref_level_db
    return normalize(S)

def deprep(S):
    S = denormalize(S)+ref_level_db
    S = librosa.db_to_power(S)
    wv = GRAD(np.expand_dims(S,0), melspecfunc, maxiter=2000, evaiter=10, tol=1e-8) ##MAXITER NORMALLY 2000 BUT SET TO 100 FOR TESTING
    print("wv shape: ", wv.shape)
    return np.array(np.squeeze(wv))

#Helper functions

#Generate spectrograms from waveform array
def tospec(data):
    print("DATA LEN:",len(data))
    specs=np.empty(len(data), dtype=object)
    for i in range(len(data)):
        x = data[i]
        S=prep(x)
        S = np.array(S, dtype=np.float32)
        specs[i]=np.expand_dims(S, -1)
    return specs

#Generate multiple spectrograms with a determined length from single wav file
def tospeclong(path, length=4*16000):
    x, sr = librosa.load(path,sr=16000)
    x,_ = librosa.effects.trim(x)
    loudls = librosa.effects.split(x, top_db=50)
    xls = np.array([])
    for interv in loudls:
        xls = np.concatenate((xls,x[interv[0]:interv[1]]))
    x = xls
    num = x.shape[0]//length
    specs=np.empty(num, dtype=object)
    for i in range(num-1):
        a = x[i*length:(i+1)*length]
        S = prep(a)
        S = np.array(S, dtype=np.float32)
        try:
            sh = S.shape
            specs[i]=S
        except AttributeError:
            print('spectrogram failed')
    return specs

#Waveform array from path of folder containing wav files
def audio_array(path):
    ls = glob(f'{path}/*.wav')
    adata = []
    for i in range(len(ls)):
        torchaudio.load(ls[i])
        x,sr = librosa.load(ls[i],sr=16000)
        # x, sr = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
        x = np.array(x, dtype=np.float32)
        adata.append(x)
    return adata

#Concatenate spectrograms in array along the time axis
def testass(a):
    but=False
    con = np.array([])
    nim = a.shape[0]
    for i in range(nim):
        im = a[i]
        im = np.squeeze(im)
        if not but:
            con=im
            but=True
        else:
            con = np.concatenate((con,im), axis=1)
    return np.squeeze(con)

#Split spectrograms in chunks with equal size
def splitcut(data):
    ls = []
    mini = 0
    minifinal = shape    #max spectrogram length
    for i in range(data.shape[0]-1):
        if data[i].shape[1]<=data[i+1].shape[1]:
            mini = data[i].shape[1]
        else:
            mini = data[i+1].shape[1]
        if mini>=3*shape and mini<minifinal:
            minifinal = mini
    for i in range(data.shape[0]):
        x = data[i]
        if x.shape[1]>=3*shape:
            for n in range(x.shape[1]//minifinal):
                ls.append(x[:,n*minifinal:n*minifinal+minifinal,:])
            ls.append(x[:,-minifinal:,:])
    return np.array(ls)


#Extract function: splitting spectrograms
def extract_image(im):
    shape = im.shape
    height = shape[2]
    width = shape[3]

    im1 = im[:,:,:, 0:(width - (2*width//3))]
    im2 = im[:,:,:, width//3:(width-(width//3))]
    im3 = im[:,:,:,(2*width//3):width]

    return im1,im2,im3

#Assemble function: concatenating spectrograms
def assemble_image(lsim):
    im1,im2,im3 = lsim
    imh = torch.cat((im1,im2,im3),dim=3)
    return imh


def tensordot_pytorch(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)

def save(ckpt_dir, netG, netD, netS, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),'netS':netS.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


def load(ckpt_dir, netG, netD, netS ,optimG,optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    netS.load_state_dict(dict_model['netS'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG,netD,netS ,optimG,optimD, epoch


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def mae(x,y):
    return torch.mean(torch.abs(x - y))

def mse(x,y):
    return torch.mean((x-y)**2)

def loss_travel(sa,sab,sa1,sab1):
    l1 = torch.mean(((sa-sa1)-(sab-sab1))**2)
    l2 = torch.mean(torch.sum(-(F.normalize(sa-sa1, p=2, dim=-1) * 
            F.normalize(sab-sab1, dim=-1)), dim=-1))
    return l1 + l2

def loss_siamese(sa,sa1):
    logits = torch.sqrt(torch.sum(((sa-sa1)**2), axis=-1, keepdim=True))
    return Variable(torch.mean(torch.max((delta - logits), 0)[0]**2), requires_grad=True)

def d_loss_f(fake):
    return Variable(torch.mean(torch.max(1 + fake, 0)[0]), requires_grad=True)

def d_loss_r(real):
    return Variable(torch.mean(torch.max(1 - real, 0)[0]), requires_grad=True)

def g_loss_f(fake):
    return torch.mean(-fake)


#Assembling generated Spectrogram chunks into final Spectrogram
def specass(a,spec):
    but=False
    con = np.array([])
    nim = a.shape[0]
    for i in range(nim-1):
        im = a[i]
        im = np.squeeze(im)
        if not but:
            con=im
            but=True
        else:
            con = np.concatenate((con,im), axis=1)
    diff = spec.shape[1]-(nim*shape)
    a = np.squeeze(a)
    con = np.concatenate((con,a[-1,:,-diff:]), axis=1)
    return np.squeeze(con)

#Splitting input spectrogram into different chunks to feed to the generator
def chopspec(spec):
    dsa=[]
    for i in range(spec.shape[1]//shape):
        im = spec[:,i*shape:i*shape+shape]
        im = np.reshape(im, (im.shape[0],im.shape[1],1))
        dsa.append(im)
    imlast = spec[:,-shape:]
    imlast = np.reshape(imlast, (imlast.shape[0],imlast.shape[1],1))
    dsa.append(imlast)
    return np.array(dsa, dtype=np.float32)

#Converting from source Spectrogram to target Spectrogram
def towave(spec, name, net, path='../content/', show=False):
    specarr = chopspec(spec)
    print(specarr.shape)
    tem = specarr
    print('Generating...')
    a = torch.Tensor(tem).permute(0,3,1,2)
    ab = net.forward(a)
    print('Assembling and Converting...')
    a = specass(a,spec)
    ab = ab.detach().numpy()
    ab = specass(ab,spec)
    awv = deprep(a)
    abwv = deprep(ab)
    print('Saving...')
    pathfin = f'{path}/{name}'
    os.mkdir(pathfin)
    sf.write(pathfin+'/AB.wav', abwv, sr)
    sf.write(pathfin+'/A.wav', awv, sr)
    print('Saved WAV!')
    IPython.display.display(IPython.display.Audio(np.squeeze(abwv), rate=sr))
    IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=sr))
    if show:
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(np.flip(a, -2), cmap=None)
        axs[0].axis('off')
        axs[0].set_title('Source')
        axs[1].imshow(np.flip(ab, -2), cmap=None)
        axs[1].axis('off')
        axs[1].set_title('Generated')
        plt.show()
    return abwv
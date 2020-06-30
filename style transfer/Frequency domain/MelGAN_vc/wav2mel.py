# -*- coding: utf-8 -*-
"""Wav2Mel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fGegbHY_mzt_3-Pw10dfUc4lbGe2eXR4
"""

!pip install soundfile
!pip install torchaudio
!pip install librosa==0.7.2

hop=192               #hop size (window size = 6*hop)
sr=22050              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

shape=32             #length of time axis of split specrograms to feed to generator            
vec_len=128           #length of vector generated by siamese vector
bs = 20               #batch size
delta = 2.            #constant for siamese loss

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
import librosa.display
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

shape=32             #length of time axis of split specrograms to feed to generator            
vec_len=128           #length of vector generated by siamese vector
bs = 20               #batch size
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
    wv = GRAD(np.expand_dims(S,0), melspecfunc, maxiter=100, evaiter=10, tol=1e-8) ##MAXITER NORMALLY 2000 BUT SET TO 100 FOR TESTING
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
        x,sr = librosa.load(ls[i],sr=22050) # sr : 16000 -> 22050
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
    minifinal = 3 * shape    #max spectrogram length
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

"""# Load Function"""

def load_wave(inst,path):    
    for i,(dirpath,dirnames,filenames) in enumerate(os.walk(path)):

        print("\n Processing :{}".format(i))
        
        for index, f in enumerate(filenames):
            file_path = os.path.join(dirpath,f)
            
            if inst in file_path:
                wav = np.load(file_path)
                wav = wav[:,np.newaxis]
                try:
                    cat = np.concatenate((cat,wav),axis=1)
                except:
                    cat = wav
    return cat

rock_other = load_wave('other','/content/drive/Shared drives/Music_Style_Transform/json_sample/rock')

rock_other = rock_other.T
rock_other_gtzan = rock_other
spec = tospec(rock_other)
rock_other_mel_gtzan = splitcut(spec)
rock_other_mel_gtzan.shape

rock_other_melon = load_wave('other','/content/drive/Shared drives/Music_Style_Transform/json_sample/melon_rock')

rock_other_melon = rock_other_melon.T
spec = tospec(rock_other_melon)
rock_other_mel_melon = splitcut(spec)
rock_other_mel_melon.shape

np.save('/content/drive/Shared drives/Music_Style_Transform/Mel_data/rock_other_mel_melon',rock_other_mel_melon)

np.save('/content/drive/Shared drives/Music_Style_Transform/Mel_data/rock_other_mel_gtzan',rock_other_mel_gtzan)

jazz_other_gtzan = load_wave('other','/content/drive/Shared drives/Music_Style_Transform/json_sample/jazz')

jazz_other_gtzan = jazz_other_gtzan.T
spec = tospec(jazz_other_gtzan)
jazz_other_mel_gtzan = splitcut(spec)
jazz_other_mel_gtzan.shape

np.save('/content/drive/Shared drives/Music_Style_Transform/Mel_data/jazz_other_mel_gtzan',jazz_other_mel_gtzan)

jazz_other_melon = load_wave('other','/content/drive/Shared drives/Music_Style_Transform/json_sample/melon_jazz')
jazz_other_melon = jazz_other_melon.T
spec = tospec(jazz_other_melon)
jazz_other_mel_melon = splitcut(spec)
jazz_other_mel_melon.shape

np.save('/content/drive/Shared drives/Music_Style_Transform/Mel_data/jazz_other_mel_melon',jazz_other_mel_melon)


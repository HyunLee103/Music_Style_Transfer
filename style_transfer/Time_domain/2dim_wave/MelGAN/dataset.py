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

from tensordot import tensordot_pytorch

## Hyper parameters

hop = 192
sr = 22050
shape = 96    # length of time axis of split spectrograms to G
 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AudioDataset(Dataset):
    def __init__(self, data, hop, shape):
        self.data = torch.Tensor(data).permute(0,3,1,2)
        self.hop = hop 
        self.shape = shape
        
    def __getitem__(self, idx):
        return self.data[idx,:,:,:].to(device)
        
    def __len__(self):
        return len(self.data)



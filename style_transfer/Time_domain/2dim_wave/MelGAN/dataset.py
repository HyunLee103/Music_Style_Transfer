import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import imageio
import librosa

from torch.autograd import Variable

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import *

class AudioDataset(Dataset):
    """
    dataset of audio waveform
    sr : 16384
    transform to 2 dimension
    """

    def __init__(self, data_dir, mode='train', direction='R2J', data_type='float32',nch=1,transform=[]):
        self.direction = direction 
        self.data_type = data_type
        self.nch = nch
        self.transform = transform
        self.mode = mode 

        if mode == 'train':
            tem_a = np.load(os.path.join(data_dir,'musdb_sample_melgan.npy'))
            tem_b = np.load(os.path.join(data_dir,'piano_cover_melgan.npy'))

            tem_a = tem_a.T
            tem_b = tem_b.T
            
            self.dataA = torch.from_numpy(tem_a).reshape(tem_a.shape[0],1,128,384)
            self.dataB = torch.from_numpy(tem_b).reshape(tem_b.shape[0],1,128,384)

        elif mode == 'test':
            tem_a = np.load(os.path.join(data_dir,'wing_30.npy'))
            # tem_b = np.load(os.path.join(data_dir,'wing_30.npy'))
     
            self.dataA = torch.from_numpy(np.flip(tem_a,axis=0).copy()).reshape(tem_a.shape[0],1,128,384)

        
    def __getitem__(self, idx):
        if self.mode == 'train':
            dataA = self.dataA[idx,:,:,:]
            dataB = self.dataB[idx,:,:,:]

            if self.direction == 'R2J':
                data = {'dataA':dataA, 'dataB':dataB}
            else:
                data = {'dataA':dataB, 'dataB':dataA}

        elif self.mode == 'test':
            dataA = self.dataA[idx,:,:,:]
            
            if self.direction == 'R2J':
                data = {'dataA':dataA}


        if self.transform:
            data = self.transform(data)

        return data

        
    def __len__(self):
        if self.mode == 'train':
            return len(self.dataB)
        elif self.mode == 'test':
            return len(self.dataA)


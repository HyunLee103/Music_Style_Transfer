import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import os
from glob import glob
import scipy

import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import os
import time
import IPython
from torch.autograd import Variable

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, direction='R2J', data_type='float32', nch=3, transform=[]):
        self.transform = transform
        self.direction = direction
        self.data_type = data_type
        self.nch = nch
        
        tem_a = np.load(os.path.join(data_dir,'melon_rock_all.npy'))
        tem_b = np.load(os.path.join(data_dir,'melon_jazz_all.npy'))

        self.dataA = torch.from_numpy(tem_a).reshape(tem_a.shape[1],3,256,256)
        self.dataB = torch.from_numpy(tem_b).reshape(tem_b.shape[1],3,256,256)


    def __getitem__(self, index):
        dataA = self.dataA[index,:,:,:]
        dataB = self.dataB[index,:,:,:]


        if self.direction == 'R2J':
            data = {'dataA': dataA, 'dataB': dataB}
        else:
            data = {'dataA': dataB, 'dataB': dataA}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.dataB)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        dataA, dataB = data['dataA'], data['dataB']

        dataA = dataA.transpose((2, 0, 1)).astype(np.float32)
        dataB = dataB.transpose((2, 0, 1)).astype(np.float32)
        return {'dataA': torch.from_numpy(dataA), 'dataB': torch.from_numpy(dataB)}


class Normalize(object):
    def __call__(self, data):
        # Nomalize [0, 1] => [-1, 1]

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data

        dataA, dataB = data['dataA'], data['dataB']
        dataA = 2 * dataA - 1
        dataB = 2 * dataB - 1
        return {'dataA': dataA, 'dataB': dataB}


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        dataA, dataB = data['dataA'], data['dataB']

        if np.random.rand() > 0.5:
            dataA = np.fliplr(dataA)
            dataB = np.fliplr(dataB)

        # if np.random.rand() > 0.5:
        #     dataA = np.flipud(dataA)
        #     dataB = np.flipud(dataB)

        return {'dataA': dataA, 'dataB': dataB}


class Rescale(object):
  """Rescale the image in a sample to a given size
  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    dataA = transform.resize(dataA, (new_h, new_w))
    dataB = transform.resize(dataB, (new_h, new_w))

    return {'dataA': dataA, 'dataB': dataB}


class RandomCrop(object):
  """Crop randomly the image in a sample
  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    dataA = dataA[top: top + new_h, left: left + new_w]
    dataB = dataB[top: top + new_h, left: left + new_w]

    return {'dataA': dataA, 'dataB': dataB}


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}


class Denomalize(object):
    def __call__(self, data):
        # Denomalize [-1, 1] => [0, 1]

        # for key, value in data:
        #     data[key] = (value + 1) / 2 * 255
        #
        # return data

        return (data + 1) / 2

        # input, label = data['input'], data['label']
        # input = (input + 1) / 2 * 255
        # label = (label + 1) / 2 * 255
        # return {'input': input, 'label': label}


















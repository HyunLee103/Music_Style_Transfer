import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

## 데이터 로더를 구현하기
class Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transforms.Compose(transform)
        
        lst_data = os.listdir(self.data_dir)
        self.lst_data_jazz = [f for f in lst_data if f.startswith('j')]
        self.lst_data_pop = [f for f in lst_data if f.startswith('p')]

    def __len__(self):
        return max(len(self.lst_data_jazz), len(self.lst_data_jazz))

    def __getitem__(self, index):
        jazz = np.load(os.path.join(self.data_dir,self.lst_data_jazz[index]))
        pop = np.load(os.path.join(self.data_dir,self.lst_data_pop[index]))
        
        if self.transform:
            jazz = self.transform(jazz)
            pop = self.transform(pop)
        
        return {'jazz': jazz, 'pop': pop}

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data



# DCGAN에 사용할 selanA image data가 DCGAN 모델의 generator output인 64x64와 맞지 않으므로
# resize 해주는 transform class 선언
class Resize(object):
    def __init__(self,shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0],self.shape[1],
                                                    self.shape[2]))
        return data

class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data['input'], data['label']
    # h, w = input.shape[:2]

    h, w = data['label'].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {'label': label, 'input': input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data




















from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import mu_law, wave_augmentation


class Dataset(Dataset):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.A_mu = np.array([mu_law(x) for x in self.A])
        self.B_mu = np.array([mu_law(x) for x in self.B])
        
    def __len__(self):
        return min(len(self.A), len(self.B))
    
    def __getitem__(self, index):
        
        return {'A': self.A_mu[index],
                'B': self.B_mu[index],
                'A_aug': mu_law(wave_augmentation(self.A[index])),
                'B_aug': mu_law(wave_augmentation(self.B[index]))}

class WaveData:
    def __init__(self, A_path, B_path):
        self.A = np.load(A_path).T
        self.B = np.load(B_path)
        
        print(f"A length: {len(self.A)}")
        print(f"B length: {len(self.B)}")
        
    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(Dataset(self.A, self.B),
                          batch_size=batch_size,
                          shuffle=shuffle)


###############################3
# for multiple domain
class WaveData:
    def __init__(self, paths):
        self.data = [np.load(path) for path in paths]
        
        for path, data in zip(paths, self.data):
            print(f"{path} length: {len(data)}")
        
    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(Dataset(self.data),
                          batch_size=batch_size,
                          shuffle=shuffle)
        
class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = max([len(d) for d in self.data])
        self.data_mu = [np.array(list(map(mu_law, data))) for data in self.data]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        out = dict()
        out_aug = dict()
        
        for i, (data, data_mu) in enumerate(zip(self.data, self.data_mu)):
            try:
                out[i] = data_mu[index]
                out_aug[i] = mu_law(wave_augmentation(data[index]))
                
            except IndexError:
                idx = index - len(data)
                out[i] = data_mu[idx]
                out_aug[i] = mu_law(wave_augmentation(data[idx]))
        
        return out, out_aug

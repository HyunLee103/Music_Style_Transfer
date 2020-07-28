from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import mu_law, wave_augmentation


class Dataset(Dataset):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        
    def __len__(self):
        return min(len(self.A), len(self.B))
    
    def __getitem__(self, index):
        
        return {'A': mu_law(self.A[index]),
                'B': mu_law(self.B[index]),
                'A_aug': mu_law(wave_augmentation(np.asfortranarray(self.A[index]))),
                'B_aug': mu_law(wave_augmentation(np.asfortranarray(self.B[index])))}

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

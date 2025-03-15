from typing import List
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class FishingAISDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray, 
                 onehot_sizes: List[int], 
                 noise_std: float):
        """
        Args:
            data (np.ndarray): 3D array of shape (num_samples, seq_len(=16, can vary), num_features(=5; fixed number: lat,lon,SOG,COG,mask)).
                              The last column is the mask.
            onehot_sizes (List[int]): list of sizes for 1-hot encodig of lat,lon,SOG,COG respectively
            noise_std (float): noise standard deviation
        """
        self.data: np.ndarray = data
        
        self.num_samples, self.seq_len, self.num_features = data.shape
        self.num_features = self.num_features - 1
        self.onehot_sizes: List[int] = onehot_sizes
        assert self.num_features == len(onehot_sizes) # should be equal to 4: lat,lon,SOG,COG without mask
        
        self.noise_std = noise_std
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Separate features and mask
        features = self.data[idx, :, :-1]  # Shape: (seq_len, 4)
        mask = self.data[idx, :, -1]       # Shape: (seq_len,)

        # One-hot encode each feature + add noise to prepare for stable diffusion
        noised_one_hot_features = []
        one_hot_features = []
        for i, onehot_size in enumerate(self.onehot_sizes):  # Loop over the features
            feature = features[:, i]  # Shape: (seq_len,)
            # Discretize using equal-width binning
            idxs = (feature * onehot_size).long().clamp(max=onehot_size - 1)
            # One-hot encode the indices
            one_hot = F.one_hot(idxs, num_classes=onehot_size).float()  # Shape: (seq_len, num_bins)

            # Add noise to masked regions
            noise = torch.randn_like(one_hot) * self.noise_std  # Gaussian noise
            mask_expanded = mask.unsqueeze(-1)  # Shape: (seq_len, 1)
            noised_one_hot = torch.where(mask_expanded == 1, one_hot + noise, one_hot)

            # Apply softmax to ensure the noised feature sums to 1
            noised_one_hot_feature = F.softmax(noised_one_hot, dim=-1)  # Shape: (seq_len, num_bins)

            one_hot_features.append(one_hot)
            noised_one_hot_features.append(noised_one_hot_feature)

        # Concatenate processed features along the last dimension
        one_hot_features = torch.cat(one_hot_features, dim=-1)  # Shape: (seq_len, sum(att_sizes))
        noised_one_hot_features = torch.cat(noised_one_hot_features, dim=-1)  # Shape: (seq_len, sum(att_sizes))
        
        input = noised_one_hot_features
        target = one_hot_features
        
        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return input, target,  mask
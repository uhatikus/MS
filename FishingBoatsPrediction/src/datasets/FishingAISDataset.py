from typing import List
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class FishingAISDataset(Dataset):
    def __init__(self, 
                 data: np.ndarray, 
                 onehot_sizes: List[int]
                 ):
        """
        Args:
            data (np.ndarray): 3D array of shape (num_samples, seq_len(=16, can vary), num_features(=5; fixed number: lat,lon,SOG,COG,mask)).
                              The last column is the mask.
            onehot_sizes (List[int]): list of sizes for 1-hot encodig of lat,lon,SOG,COG respectively
        """
        super().__init__()
        self.data: np.ndarray = data
        
        self.num_samples, self.seq_len, self.num_features = data.shape
        self.num_features = self.num_features - 1
        self.onehot_sizes: List[int] = onehot_sizes
        assert self.num_features == len(onehot_sizes) # should be equal to 4: lat,lon,SOG,COG without mask
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Separate features and mask
        features: np.ndarray = self.data[idx, :, :-1]  # Shape: (seq_len, 4)
        mask: np.ndarray = self.data[idx, :, -1]       # Shape: (seq_len, )

        # One-hot encode each feature
        one_hot_features = []
        for i, (feature, size) in enumerate(zip(features.T, self.onehot_sizes)):# Loop over the features
            feature = torch.from_numpy(feature).float() # Shape: (seq_len,)
            # Discretize using equal-width binning
            idxs = (feature * (size - 1)).long()
            # One-hot encode the indices
            one_hot: torch.Tensor = F.one_hot(idxs, num_classes=size).float()  # Shape: (seq_len, num_bins)
            one_hot_features.append(one_hot)

        # concatanate all features
        one_hot_features = torch.cat(one_hot_features, dim=-1) # Shape: (seq_len, all_one_hot_features_len)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).float()
        
        return one_hot_features, mask
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
    
    @staticmethod
    def one_hot_to_continuous(one_hot_features: torch.Tensor, 
                              masks: torch.Tensor,
                              onehot_sizes: List[int]
                              ) -> np.ndarray:
        """
        Convert one-hot encoded features back to continuous values.
        
        Args:
            one_hot_features (torch.Tensor): One-hot encoded features of shape (samples_len, seq_len, sum(onehot_sizes))
            masks (torch.Tensor): masks of shape (samples_len, seq_len)
            onehot_sizes (List[int]): List of sizes for 1-hot encoding of each feature (lat, lon, SOG, COG)
            
        Returns:
            np.ndarray: Continuous features array of shape (seq_len, len(onehot_sizes))
        """
        # Split the concatenated one-hot features into individual features
        split_features = torch.split(one_hot_features, onehot_sizes, dim=-1)
        
        # Convert each one-hot feature back to continuous value
        continuous_features = []
        for i, (feature, size) in enumerate(zip(split_features, onehot_sizes)):
            # Get the indices of the maximum values (argmax)
            indices = torch.argmax(feature, dim=-1).float()  # Shape: (samples_len, seq_len)
            
            # Convert indices back to continuous values in [0, 1] range
            continuous = indices / (size - 1)
            
            continuous_features.append(continuous.unsqueeze(-1))  # Add feature dimension
        
        # Concatenate all features along the feature dimension
        continuous_features = torch.cat(continuous_features, dim=-1)  # Shape: (samples_len, seq_len, num_features)
        
        # Add masks as 5th column
        combined = torch.cat([
            continuous_features, 
            masks.unsqueeze(-1)
        ], dim=-1)
        
        return combined
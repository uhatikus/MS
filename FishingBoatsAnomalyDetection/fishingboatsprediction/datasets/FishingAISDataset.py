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
                            onehot_sizes: List[int],
                            threshold: float = 0.7
                            ) -> torch.Tensor:
        """
        Convert one-hot encoded features back to continuous values using threshold-based averaging.
        For each feature, it takes the average of all indices whose one-hot value is at least the given threshold.
        If none meet the threshold, the value is set to NaN, which is later filled via linear interpolation along the sequence.
        
        Args:
            one_hot_features (torch.Tensor): One-hot encoded features of shape (samples_len, seq_len, sum(onehot_sizes))
            masks (torch.Tensor): Masks of shape (samples_len, seq_len)
            onehot_sizes (List[int]): List of sizes for one-hot encoding of each feature
            threshold (float): The threshold above which an index is considered active
            
        Returns:
            torch.Tensor: Continuous features array of shape (samples_len, seq_len, len(onehot_sizes)+1)
                        where the last column is the mask.
        """
        # Split the concatenated one-hot features into individual features.
        split_features = torch.split(one_hot_features, onehot_sizes, dim=-1)

        continuous_features = []

        # Process each feature individually.
        for size, feature in zip(onehot_sizes, split_features):
            # Create an index tensor for the feature dimension.
            index_tensor = torch.arange(size, device=feature.device).float()  # Shape: (size,)
            
            # Determine which indices have values above the threshold.
            above_threshold = feature >= threshold  # Shape: (batch_size, seq_len, size)
            
            # Compute the weighted sum of indices (only those above the threshold) and count the number of active ones.
            # Broadcasting index_tensor over the last dimension.
            weighted_sum = (above_threshold * index_tensor).sum(dim=-1)  # (batch_size, seq_len)
            count = above_threshold.sum(dim=-1)  # (batch_size, seq_len)
            
            # Calculate the average index, setting locations with no valid indices to NaN.
            avg_index = torch.where(
                count == 0,
                torch.full_like(weighted_sum, float('nan')),
                weighted_sum / count
            )
            
            # Convert the average index to the [0, 1] range.
            continuous = avg_index / (size - 1)
            continuous_features.append(continuous.unsqueeze(-1))  # Shape becomes (batch_size, seq_len, 1)
        
        # Concatenate all features along the last dimension.
        continuous_features = torch.cat(continuous_features, dim=-1)  # Shape: (batch_size, seq_len, num_features)
        
        # Define a helper function that does linear interpolation for each sample and each feature.
        def interpolate_nans(x: torch.Tensor) -> torch.Tensor:
            """
            For each sample and each feature column in x (assumed shape: [batch, seq_len, features]),
            linearly interpolate any NaN values along the sequence dimension.
            """
            B, T, F = x.shape
            # Process each sample and feature individually.
            for b in range(B):
                for f in range(F):
                    y = x[b, :, f]
                    # Get indices of valid (non-NaN) entries.
                    valid_mask = ~torch.isnan(y)
                    if valid_mask.sum() == 0:
                        # No valid entries, skip interpolation.
                        continue
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0].float()  # indices where value is not nan
                    # For each time step, perform interpolation if necessary.
                    for t in range(T):
                        if torch.isnan(y[t]):
                            # Find the closest valid indices before and after t.
                            left_indices = valid_indices[valid_indices < t]
                            right_indices = valid_indices[valid_indices > t]
                            
                            if left_indices.numel() > 0 and right_indices.numel() > 0:
                                left_idx = int(left_indices[-1].item())
                                right_idx = int(right_indices[0].item())
                                # Compute weight based on relative distance.
                                weight = (t - left_idx) / (right_idx - left_idx)
                                y[t] = y[left_idx] * (1 - weight) + y[right_idx] * weight
                            elif left_indices.numel() > 0:
                                # If only left valid exists, use its value.
                                left_idx = int(left_indices[-1].item())
                                y[t] = y[left_idx]
                            elif right_indices.numel() > 0:
                                # If only right valid exists, use its value.
                                right_idx = int(right_indices[0].item())
                                y[t] = y[right_idx]
                    x[b, :, f] = y
            return x

        # Interpolate any NaN values in continuous_features.
        continuous_features = interpolate_nans(continuous_features)
        
        # Append the masks as the final column.
        combined = torch.cat([
            continuous_features,
            masks.unsqueeze(-1)
        ], dim=-1)
        
        return combined

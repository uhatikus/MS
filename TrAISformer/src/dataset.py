import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class AISDataset(Dataset):
    """Customized Pytorch dataset."""

    def __init__(self, filename, max_seqlen=144, device=torch.device("cpu")):
        """
        Args
            filename: path to the pickle file with thie list of dictionaries,
            where each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP, MMSI]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is 144.
            device: (optional) torch.device("cpu") or torch.device("cuda:0")
        """
        self.filename = filename
        self.device = device
        self.max_seqlen = max_seqlen

        with open(self.filename, "rb") as f:
            self.data = pickle.load(f)

        self.remove_nans()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
        """
        data_item = self.data[idx]

        # get trahectory
        trajectory = data_item["traj"][:, :4]  # lat, lon, sog, cog

        seqlen = min(len(trajectory), self.max_seqlen)

        # trajectory sequence
        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = trajectory[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        # mask of the trajectory sequence
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.0

        # seqlen = torch.tensor(seqlen, dtype=torch.int)
        # mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        # time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, mask  # , seqlen, mmsi, time_start

    def remove_nans(self):
        self.data = [x for x in self.data if not np.isnan(x["traj"]).any()]

    def get_current_max_seqlen(self):
        max_len = 0
        for i in range(len(self.data)):
            if len(self.data[i]["traj"]) > max_len:
                max_len = len(self.data[i]["traj"])
        return max_len  # should be 144 or less

    def get_current_min_seqlen(self):
        min_len = 100000
        for i in range(len(self.data)):
            if len(self.data[i]["traj"]) < min_len:
                min_len = len(self.data[i]["traj"])
                min_i = i
        return min_len  # should be more than 24

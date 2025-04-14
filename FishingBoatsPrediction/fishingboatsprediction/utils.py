from typing import Dict, List
import pickle
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from fishingboatsprediction.configs.configs import DefaultConfig
from fishingboatsprediction.datasets.FishingAISDataset import FishingAISDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def get_dataloaders(config: DefaultConfig, phases: List[str] = ["train", "validation", "test"]) -> Dict[str, DataLoader]:  
    dataloaders: Dict[DataLoader] = {}  
    
    print("Dataset Information:")
    print("-" * 50)
    
    for phase in phases:
        datapath = os.path.join(config.dataset_dir, f"X_{phase}.pkl")
        with open(datapath, "rb") as f:
            data = pickle.load(f)
        dataset = FishingAISDataset(data, config.onehot_sizes)
        dataloaders[phase] = DataLoader(dataset,
                                        batch_size=config.batch_size,
                                        shuffle = (phase == "train"))
        
        print(f"{phase.capitalize()} Dataset:")
        print(f"\tNumber of samples: {len(dataset)}")
        print(f"\tBatch size: {config.batch_size}")
        print(f"\tNumber of batches: {len(dataloaders[phase])}")
        if hasattr(dataset, '__getitem__') and len(dataset) > 0:
            one_hot_feature_sample, mask_sample = dataset[0]
            print(f"\tOne_hot_feature_sample size: {one_hot_feature_sample.size()}")
            print(f"\tMask_sample size: {mask_sample.size()}")
        print("-" * 50)
        
    return dataloaders
from typing import Dict
from torch.utils.data import DataLoader

# from clearml import Task

import pickle
import os

from configs.configs import DefaultConfig
from datasets.FishingBoatsDataset import FishingAISDataset, AISPreprocessor
from models.AIStableDiffusion import AISNoiseScheduler
from trainers.AIStableDiffusionTrainer import AIStableDiffusionTrainer
from utils import set_seed
from models import AISUNet

set_seed(42)

def get_dataloaders(config: DefaultConfig) -> Dict[str, DataLoader]:  
    dataloaders: Dict[DataLoader] = {}  
    phases = ["train", "validation", "test"]
    
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

def run_experiment_with_clearml():
    config: DefaultConfig = DefaultConfig()
    # Initialize a ClearML task
    # task = Task.init(project_name="Fishing Boats: AIStable Diffusion", task_name="Test 0")
    # task.connect(config)
    
    ## Data
    # ===============================
    # aisp = AISPreprocessor(dataset_path = 'data/fishing_boats_dynamic/Dynamic_*.csv')
    # aisp.run()

    dataloaders: Dict[DataLoader] = get_dataloaders(config)
    
    ## Model and Scheduler
    # ===============================
    model: AISUNet = AISUNet(config).to(config.device)
    scheduler: AISNoiseScheduler = AISNoiseScheduler(config)
    
    ## Trainer
    # ===============================
    trainer: AIStableDiffusionTrainer = AIStableDiffusionTrainer(config, model, scheduler, dataloaders)
    
    ## Training
    # ===============================
    trainer.train()
    
    ## Testing
    # ===============================
    trainer.generate_test_samples()
            
    # Log metrics
    # task.get_logger().report_scalar("Loss", "train", value=0.1, iteration=1)
    # task.get_logger().report_scalar("Accuracy", "train", value=0.95, iteration=1)


if __name__ == "__main__":
    run_experiment_with_clearml()
from torch.utils.data import DataLoader
from clearml import Task

import pickle
import os

from src.configs.configs import DefaultConfig
from src.datasets.FishingBoatsDataset import FishingAISDataset
from src.trainers.AIStableDiffusionTrainer import AIStableDiffusionTrainer
from utils import set_seed
from src.models import AISUNet

set_seed(42)

def get_dataloaders(config: DefaultConfig):  
    dataloaders = {}  
    phases = ["train", "validation", "test"]
    for phase in phases:
        datapath = os.path.join(config.dataset_dir, f"X_{phase}.pkl")
        with open(datapath, "rb") as f:
            data = pickle.load(f)
        dataset = FishingAISDataset(data, config.onehot_sizes)
        dataloaders[phase] = DataLoader(dataset,
                                        batch_size=config.batch_size,
                                        shuffle = (phase == "train"))
    return dataloaders

def run_experiment_with_clearml():
    config: DefaultConfig = DefaultConfig()
    # Initialize a ClearML task
    task = Task.init(project_name="Fishing Boats: AIStable Diffusion", task_name="Test 0")
    task.connect(config)
    
    ## Data
    # ===============================
    dataloaders = get_dataloaders(config)
    
    ## Model
    # ===============================
    model = AISUNet(config)
    
    ## Trainer
    # ===============================
    trainer = AIStableDiffusionTrainer(config, model, dataloaders)
    
    ## Training
    # ===============================
    trainer.train()
            
    # Log metrics
    task.get_logger().report_scalar("Loss", "train", value=0.1, iteration=1)
    task.get_logger().report_scalar("Accuracy", "train", value=0.95, iteration=1)


if __name__ == "__main__":
    run_experiment_with_clearml()
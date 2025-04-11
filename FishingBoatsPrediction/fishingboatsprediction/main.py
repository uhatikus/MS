from typing import Dict
import pickle
import os

from clearml import Task
from torch.utils.data import DataLoader

from configs.configs import DefaultConfig, TestConfig
from datasets.FishingAISDataset import FishingAISDataset
from datasets.AISPreprocessor import AISPreprocessor
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
    # config: DefaultConfig = DefaultConfig()
    print("Reading config...")
    config: DefaultConfig = TestConfig()
    print("Config is ready.")
    # Initialize a ClearML task
    clearml_task: Task = Task.init(
        project_name="Fishing Boats: AIStable Diffusion",
        task_name="Test 0",
        # Add these for better reproducibility:
        output_uri=True,  # Auto-upload artifacts to ClearML server
        auto_connect_frameworks=True,  # Auto-log PyTorch/TensorFlow/Keras
    )
    clearml_task.connect(config, name="config")  # Explicitly name the config
    
    
    ## Data
    # ===============================
    if not os.path.exists(config.dataset_dir):
        print("Creating new dataset..")
        aisp = AISPreprocessor(dataset_path = 'data/FishingKoreaAIS/Dynamic_*.csv',
                                target_freq_in_minutes = config.target_freq_in_minutes,
                                trajectory_sequence_len = config.trajectory_sequence_len,
                                max_trajectory_sequence_len_to_predict = config.max_trajectory_sequence_len_to_predict,
                                min_trajectory_sequence_len_to_predict = config.min_trajectory_sequence_len_to_predict,
                                rotate_trajetories = config.rotate_trajetories,
                                synthetic_ratio = config.synthetic_ratio,
                                prediction_buffer = config.prediction_buffer, 
                                validation_size_ratio = config.validation_size_ratio, # from 0 to 1
                                test_size_ratio = config.test_size_ratio, # from 0 to 1
                                output_dir = config.output_dir,
                                dataset_dir = config.dataset_dir)
        aisp.run()
    else:
        print(f"Dataset {config.dataset_dir} is already created.")

    dataloaders: Dict[DataLoader] = get_dataloaders(config)
    
    ## Model and Scheduler
    # ===============================
    model: AISUNet = AISUNet(config).to(config.device)
    scheduler: AISNoiseScheduler = AISNoiseScheduler(config)
    
    ## Trainer
    # ===============================
    trainer: AIStableDiffusionTrainer = AIStableDiffusionTrainer(config, model, scheduler, dataloaders, clearml_task)
    
    ## Training
    # ===============================
    trainer.train()
    
    ## Testing
    # ===============================
    trainer.generate_test_samples()

if __name__ == "__main__":
    run_experiment_with_clearml()
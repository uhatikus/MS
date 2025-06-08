from typing import Dict
import os
import pickle
import shutil

from clearml import Task
from torch.utils.data import DataLoader

from fishingboatsprediction.configs.configs import DefaultConfig, TestConfig
from fishingboatsprediction.datasets.AISPreprocessor import AISPreprocessor
from fishingboatsprediction.models.AIStableDiffusion import AISNoiseScheduler
from fishingboatsprediction.trainers.AIStableDiffusionTrainer import AIStableDiffusionTrainer
from fishingboatsprediction.utils import set_seed, get_dataloaders
from fishingboatsprediction.models import AISUNet

set_seed(42)

def run_experiment_with_clearml():
    clearml_task: Task = Task.init(
        project_name="Fishing Boats: AIStable Diffusion",
        task_name="Test 2",
        # Add these for better reproducibility:
        output_uri=True,  # Auto-upload artifacts to ClearML server
        auto_connect_frameworks=True,  # Auto-log PyTorch/TensorFlow/Keras
    )
    
    run_experiment(clearml_task)
    
def run_experiment(clearml_task: Task | None  = None):
    if clearml_task is None:
        print("Running the experiment without ClearML")
    else:
        print("Running the experiment WITH ClearML")
    print("Reading config...")
    config: DefaultConfig = DefaultConfig()
    # config: DefaultConfig = TestConfig()
    print("Config is ready.")
    
    if clearml_task is not None:
        clearml_task.connect(config, name="config")  # Explicitly name the config
        with open(f'{config.output_dir}/metadata.txt', 'w') as f:   
            info = f"""{clearml_task.get_output_log_web_page()}
            """
            f.write(info)
       
    
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
    trainer.generate_test_samples(only_one=True) # only_one=True to create only one sample for testing purposes
    
def test_model(model_dir, model_name):
    config_path=f"{model_dir}/config.yaml"
    config: DefaultConfig = DefaultConfig(config_path)
    dataloaders: Dict[DataLoader] = get_dataloaders(config, phases=["test"])

    ## Model and Scheduler
    # ===============================
    model: AISUNet = AISUNet(config).to(config.device)
    scheduler: AISNoiseScheduler = AISNoiseScheduler(config)

    # Load the trained model weights
    model_path = f"{model_dir}/{model_name}"
    model.load(model_path)
    
    # copy model, just in case
    shutil.copy2(model_path, f"{config.output_dir}/{model_name}")

    ## Trainer
    # ===============================
    trainer: AIStableDiffusionTrainer = AIStableDiffusionTrainer(config, model, scheduler, dataloaders)

    ## Testing
    # ===============================
    
    datapath = os.path.join(config.dataset_dir, "X_test.pkl")
    with open(datapath, "rb") as f:
        original_samples = pickle.load(f)
        
    list_of_reconstructed_samples_np = []
    N = 20
    for i in range(N):
        _, reconstructed_samples_np = trainer.generate_test_samples(only_one=True)
        list_of_reconstructed_samples_np.append(reconstructed_samples_np)

    
    # AISPreprocessor.plot_individual_trajectory_comparisons(original_samples, reconstructed_samples_np, output_dir=config.output_dir)
    AISPreprocessor.plot_probabilistic_trajectory_comparisons(original_samples, list_of_reconstructed_samples_np, output_dir=config.output_dir)
    
def continue_experiment_with_clearml(model_dir: str, model_name: str):
    clearml_task: Task = Task.init(
        project_name="Fishing Boats: AIStable Diffusion",
        task_name="Continue Test 2",
        # Add these for better reproducibility:
        output_uri=True,  # Auto-upload artifacts to ClearML server
        auto_connect_frameworks=True,  # Auto-log PyTorch/TensorFlow/Keras
    )
    
    continue_experiment(model_dir, model_name, clearml_task)
    
def continue_experiment(model_dir: str, model_name: str, clearml_task: Task | None  = None):
    if clearml_task is None:
        print("Running the experiment without ClearML")
    else:
        print("Running the experiment WITH ClearML")
    print("Reading config...")
    
    # Load config from the existing experiment
    config_path = f"{model_dir}/config.yaml"
    config: DefaultConfig = DefaultConfig(config_path)
    print("Config is ready.")
    
    # Connect config to ClearML if using it
    if clearml_task is not None:
        clearml_task.connect(config, name="config")  # Explicitly name the config
        with open(f'{config.output_dir}/metadata.txt', 'w') as f:   
            info = f"""{clearml_task.get_output_log_web_page()}
            """
            f.write(info)
    
    # Get dataloaders
    dataloaders: Dict[DataLoader] = get_dataloaders(config)
    
    # Initialize model and scheduler
    model: AISUNet = AISUNet(config).to(config.device)
    scheduler: AISNoiseScheduler = AISNoiseScheduler(config)
    
    # Load the existing model weights
    model_path = f"{model_dir}/{model_name}"
    model.load(model_path)
    print(f"Loaded model weights from {model_path}")
    
    # Initialize trainer
    trainer: AIStableDiffusionTrainer = AIStableDiffusionTrainer(
        config, model, scheduler, dataloaders, clearml_task
    )
    
    # Continue training
    trainer.train()
    
    # Generate test samples
    trainer.generate_test_samples(only_one=True)
    
if __name__ == "__main__":
    print("Start")
    # run_experiment()
    # run_experiment_with_clearml()
    test_model(model_dir="results/test_2/20250415_084414", model_name="best_model_47.pth")
    # test_model(model_dir="results/test_1/20250414_192606", model_name="best_model_1.pth")
    # continue_experiment(model_dir="results/test_1/20250414_192606", model_name="best_model_1.pth")
    # continue_experiment_with_clearml(model_dir="results/test_1/20250414_192606", model_name="best_model_1.pth")
    print("start")
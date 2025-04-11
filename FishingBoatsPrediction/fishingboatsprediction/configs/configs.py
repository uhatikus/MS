import torch
import yaml
from datetime import datetime

class DefaultConfig:
    def __init__(self, config_path = "fishingboatsprediction/configs/default_config.yaml"):
        self.devices = {
            "cpu": torch.device("cpu"),
            "gpu": torch.device("cuda:0")
        }
        
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            
        self.device =  self.devices[data["device"]]
        
        # AISPreprocessor
        self.target_freq_in_minutes: int = data["target_freq_in_minutes"]
        self.trajectory_sequence_len: int = data["trajectory_sequence_len"]
        self.max_trajectory_sequence_len_to_predict: int = data["max_trajectory_sequence_len_to_predict"]
        self.min_trajectory_sequence_len_to_predict: int = data["min_trajectory_sequence_len_to_predict"]
        self.rotate_trajetories: bool = data["rotate_trajetories"]
        # self.shitf_trajectories: bool = data["shitf_trajectories"]
        # self.cols: AISColumnNames = AISColumnNames(),
        self.synthetic_ratio: float = data["synthetic_ratio"]
        self.prediction_buffer: int = data["prediction_buffer"]
        
        self.validation_size_ratio: float = data["validation_size_ratio"] # from 0 to 1
        self.test_size_ratio: float = data["test_size_ratio"] # from 0 to 1

        self.dataset_dir = f"data/datasets/dataset_{self.target_freq_in_minutes}_{self.trajectory_sequence_len}_{self.max_trajectory_sequence_len_to_predict}_{self.min_trajectory_sequence_len_to_predict}_{self.rotate_trajetories}_{self.synthetic_ratio}_{self.prediction_buffer}_{self.validation_size_ratio}_{self.test_size_ratio}"
        
        # FishingAISDataset
        self.lat_size = data["lat_size"]
        self.lon_size = data["lon_size"]
        self.sog_size = data["sog_size"]
        self.cog_size = data["cog_size"]
        self.onehot_sizes = [self.lat_size, self.lon_size, self.sog_size, self.cog_size]
        assert sum(self.onehot_sizes) % 8 == 0
        
        # AISUNet
        self.base_channels = data["base_channels"]
        self.time_dim = data["time_dim"]
        
        # AISNoiseScheduler
        self.num_timesteps = data["num_timesteps"]
        self.beta_start = data["beta_start"]
        self.beta_end = data["beta_end"]
        self.schedule = data["schedule"]
        
        # AIStableDiffusionTrainer
        self.learning_rate = data["learning_rate"]
        self.weight_decay = data["weight_decay"]
        self.num_epochs = data["num_epochs"]
        self.eta_min = data["eta_min"]
        self.batch_size = data["batch_size"]
        self.use_snr_weighting = data["use_snr_weighting"]
        self.log_interval = data["log_interval"]
        self.noise_magnitude = data["noise_magnitude"]
        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{data["output_dir"]}/{timestamp}"

class TestConfig(DefaultConfig):
    def __init__(self, config_path="fishingboatsprediction/configs/test_config.yaml"):
        super().__init__(config_path)
        

class ProdConfig(DefaultConfig):
    def __init__(self, config_path="fishingboatsprediction/configs/prod_config.yaml"):
        super().__init__(config_path)
        
        
if __name__ == "__main__":     
    dc = DefaultConfig()
    print(dc.device)
import torch
import yaml

class DefaultConfig:
    def __init__(self, config_path = "src/configs/default_config.yaml"):
        self.devices = {
            "cpu": torch.device("cpu"),
            "gpu": torch.device("cuda:0")
        }
        
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            
        self.device =  self.devices[data["device"]]
        
        # FishingAISDataset
        self.dataset_dir = data["dataset_dir"]
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
        self.output_dir = data["output_dir"]
        self.noise_magnitude = data["noise_magnitude"]

        
dc = DefaultConfig()
print(dc.device)
import torch
import yaml

class DefaultConfig:
    def __init__(self, config_path = "src/default_config.yaml"):
        self.devices = {
            "cpu": torch.device("cpu"),
            "gpu": torch.device("cuda:0")
        }
        
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            

        self.device =  self.devices[data["device"]]
        self.dataset_dir = data["dataset_dir"]
        
        self.lat_size = data["lat_size"]
        self.lon_size = data["lon_size"]
        self.sog_size = data["sog_size"]
        self.cog_size = data["cog_size"]
        self.onehot_sizes = [self.lat_size, self.lon_size, self.sog_size, self.cog_size]
        
        self.max_epochs = data["max_epochs"]
        self.batch_size = data["batch_size"]
        self.learning_rate = data["learning_rate"]
        
        self.retrain = data["retrain"]
        
        self.n_samples = data["n_samples"]
        
        
dc = DefaultConfig()
print(dc.device)
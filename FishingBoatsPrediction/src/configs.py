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
            
        self.retrain = data["retrain"]
        self.device =  self.devices[data["device"]]
        self.max_epochs = data["max_epochs"]
        self.batch_size = data["batch_size"]
        self.n_samples = data["n_samples"]
        self.learning_rate = data["learning_rate"]
        
dc = DefaultConfig()
print(dc.device)
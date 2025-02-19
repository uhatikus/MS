
import torch 
import torch.nn as nn

class ConditionedDiffusionModel(nn.Module):
    def __init__(self, time_steps=100):
        super().__init__()
        self.time_steps = time_steps
        self.linear = nn.Sequential(
            nn.Linear(6, 128),  # Input: (x, y, sog, cog, start_x, start_y)
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: (x, y, sog, cog)
        )
    
    def forward(self, x, start_end, t):
        return self.linear(torch.cat([x, start_end, t.unsqueeze(-1)], dim=-1))
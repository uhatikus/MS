import torch
import torch.nn as nn
import torch.nn.functional as F

class AISUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        self.time_embed = nn.Linear(1, base_channels)
        
        # Downsample
        self.down1 = self._block(in_channels, base_channels)
        self.down2 = self._block(base_channels, base_channels*2)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels*2, base_channels*4)
        
        # Upsample
        self.up1 = self._block(base_channels*4 + base_channels*2, base_channels*2)
        self.up2 = self._block(base_channels*2 + base_channels, base_channels)
        self.conv_last = nn.Conv2d(base_channels, out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.view(-1, 1)).unsqueeze(-1).unsqueeze(-1)
        
        # Downsample path
        x1 = self.down1(x) + t_emb
        x = self.pool(x1)
        x2 = self.down2(x) + t_emb
        x = self.pool(x2)
        
        # Bottleneck
        x = self.bottleneck(x) + t_emb
        
        # Upsample path
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x) + t_emb
        
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x) + t_emb
        
        return self.conv_last(x)
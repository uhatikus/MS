import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.configs import DefaultConfig

class TimeEmbedding(nn.Module):
    """Improved time embedding with sinusoidal positional encoding"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class AISUNet(nn.Module):
    """UNet for 2D matrix denoising with time embedding"""
    def __init__(self, 
                 config: DefaultConfig):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(config.time_dim),
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
        )

        # Time projections to match channel dims at each block
        self.time_proj1 = nn.Linear(config.time_dim, config.base_channels)
        self.time_proj2 = nn.Linear(config.time_dim, config.base_channels * 2)
        self.time_proj3 = nn.Linear(config.time_dim, config.base_channels * 4)
        self.time_proj_bottleneck = nn.Linear(config.time_dim, config.base_channels * 8)
        self.time_proj_up1 = nn.Linear(config.time_dim, config.base_channels * 4)
        self.time_proj_up2 = nn.Linear(config.time_dim, config.base_channels * 2)
        self.time_proj_up3 = nn.Linear(config.time_dim, config.base_channels)

        # Encoder blocks
        self.down1 = self._block(1, config.base_channels)
        self.down2 = self._block(config.base_channels, config.base_channels * 2)
        self.down3 = self._block(config.base_channels * 2, config.base_channels * 4)

        # Bottleneck
        self.bottleneck = self._block(config.base_channels * 4, config.base_channels * 8)

        # Decoder blocks
        self.up1 = self._block(config.base_channels * 12, config.base_channels * 4)
        self.up2 = self._block(config.base_channels * 6, config.base_channels * 2)
        self.up3 = self._block(config.base_channels * 3, config.base_channels)

        # Final convolution
        self.out_conv = nn.Conv2d(config.base_channels, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        
        # # To split the output into parts and apply softmax to each part
        # self.lat_start = 0
        # self.lat_end = config.lat_size
        # self.lon_end = self.lat_end + config.lon_size
        # self.sog_end = self.lon_end + config.sog_size
        # self.cog_end = self.sog_end + config.cog_size

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def inject_time(self, x, t_emb, proj):
        t_proj = proj(t_emb)
        return x + t_proj[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: [B, seq_len, one_hot_featuures_len]
        # Reshape to [B, 1, seq_len, one_hot_features_len] to treat as a single-channel image
        x = x.unsqueeze(1)  # [B, 1, seq_len, one_hot_features_len]
        
        t_emb = self.time_embed(t)

        x1 = self.down1(x)
        x1 = self.inject_time(x1, t_emb, self.time_proj1)

        x2 = self.down2(self.pool(x1))
        x2 = self.inject_time(x2, t_emb, self.time_proj2)

        x3 = self.down3(self.pool(x2))
        x3 = self.inject_time(x3, t_emb, self.time_proj3)

        x4 = self.bottleneck(self.pool(x3))
        x4 = self.inject_time(x4, t_emb, self.time_proj_bottleneck)

        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        x = self.inject_time(x, t_emb, self.time_proj_up1)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.inject_time(x, t_emb, self.time_proj_up2)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)
        x = self.inject_time(x, t_emb, self.time_proj_up3)

        output = self.out_conv(x)
        
        # Remove channel dimension to return [B, seq_len, one_hot_feature_len]
        output = output.squeeze(1)
        
        # # Split the tensor
        # lat_part = output[:, :, self.lat_start:self.lat_end]
        # lon_part = output[:, :, self.lat_end:self.lon_end]
        # sog_part = output[:, :, self.lon_end:self.sog_end]
        # cog_part = output[:, :, self.sog_end:self.cog_end]
        
        # # Apply softmax to each part along the feature dimension
        # lat_part = F.softmax(lat_part, dim=-1)
        # lon_part = F.softmax(lon_part, dim=-1)
        # sog_part = F.softmax(sog_part, dim=-1)
        # cog_part = F.softmax(cog_part, dim=-1)
        
        # # Concatenate the parts back together
        # output = torch.cat([lat_part, lon_part, sog_part, cog_part], dim=-1)
        
        return output
    
class AISNoiseScheduler:
    """Improved noise scheduler with cosine schedule option"""
    def __init__(self, 
                 config: DefaultConfig):
        
        self.num_timesteps = config.num_timesteps
        self.schedule = config.schedule
        
        if config.schedule == 'linear':
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        elif config.schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            s = 0.008
            steps = config.num_timesteps + 1
            x = torch.linspace(0, config.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / config.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
            
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, 
                one_hot_features: torch.Tensor, 
                mask: torch.Tensor, 
                noise: torch.Tensor, 
                timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timesteps])
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[timesteps])
        
        # Reshape for broadcasting with (batch_size, seq_len, features)
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]
        
        # Expand mask to match one_hot_features dimensions
        mask_expanded = mask.unsqueeze(-1)  # Shape: [batch_size, 16, 1]
        
        # Add noise to masked regions
        noised_one_hot_features = torch.where(mask_expanded == 1,
                                        sqrt_alpha_bar * one_hot_features + sqrt_one_minus_alpha_bar * noise,
                                        one_hot_features)

        return noised_one_hot_features

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def get_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get weights for loss based on timesteps (SNR weighting)"""
        snr = self.alpha_bars[timesteps] / (1 - self.alpha_bars[timesteps])
        loss_weights = torch.sqrt(1.0 / (snr + 1e-8))
        return loss_weights
    
    def step(self, 
             x: torch.Tensor, 
             mask: torch.Tensor,
             noise_pred: torch.Tensor, 
             t: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion step (denoising)
        
        Args:
            x: Current noisy sample [batch, seq_len, features]
            mask: Binary mask indicating regions to denoise [batch, seq_len]
            noise_pred: Model's prediction of the noise [batch, seq_len, features]
            t: Current timestep for each sample in batch [batch]
            
        Returns:
            Denoised sample (only for masked regions)
        """
        # Expand dimensions for broadcasting
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)        # [batch, 1, 1]
        alpha_bar_prev = self.alpha_bars[t-1].view(-1, 1, 1) if t.min() > 0 else torch.ones_like(alpha_bar)
        
        # Algorithm 2 from DDPM paper with mask support:
        # 1. Predict the original sample
        pred_original = (x - noise_pred * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
        
        # 2. Compute direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_bar_prev) * noise_pred
        
        # 3. Combine these for the final prediction
        x_prev = torch.sqrt(alpha_bar_prev) * pred_original + pred_dir
        
        # Only update masked regions
        mask_expanded = mask.unsqueeze(-1)
        
        x = x_prev * mask_expanded
        
        return x
from typing import Dict
import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

from clearml import Task, Logger

from configs.configs import DefaultConfig
from models.AIStableDiffusion import AISNoiseScheduler, AISUNet
from fishingboatsprediction.datasets.FishingAISDataset import FishingAISDataset

class AIStableDiffusionTrainer:
    def __init__(self, config: DefaultConfig, model: AISUNet, scheduler: AISNoiseScheduler, dataloaders: Dict[str, DataLoader], clearml_task: Task | None = None):
        self.config: DefaultConfig = config
        self.model: AISUNet = model
        self.scheduler: AISNoiseScheduler = scheduler
        self.dataloaders: Dict[str, DataLoader] = dataloaders
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
        
        self.clearml_task: Task | None = clearml_task
        self.logger: Logger | None = None
        if self.clearml_task is not None:
            self.logger: Logger = self.clearml_task.get_logger()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def train(self):
        """Complete training loop"""
        best_validation_loss = float('inf')
        train_losses = []
        validation_losses = []
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training epoch
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validation
            validation_loss = self._validate(epoch)
            validation_losses.append(validation_loss)
            
            # Save best model
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self._save_checkpoint(epoch, validation_loss, f'best_model_{epoch}.pth')
            
            if self.clearml_task is not None:
                logger = self.clearml_task.get_logger()
                logger.report_scalar(
                    title="Training Loss", 
                    series="Batch Loss", 
                    value=train_loss, 
                    iteration=epoch
                )
                logger.report_scalar(
                    title="Validation Loss", 
                    series="Batch Loss", 
                    value=validation_loss, 
                    iteration=epoch
                )
                # Log LR (since you're using a scheduler)
                logger.report_scalar(
                    title="Learning Rate", 
                    series="LR", 
                    value=self.optimizer.param_groups[0]['lr'], 
                    iteration=epoch
                )
        # Plot losses
        self._plot_losses(train_losses, validation_losses)
                
    def _train_epoch(self, epoch):
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0
        
        # print(f"dataloader: {self.dataloaders["train"].batch_size}")
        for batch_idx, (one_hot_features, masks) in enumerate(self.dataloaders["train"]):
            # Move data to device
            one_hot_features = one_hot_features.to(self.config.device)
            masks = masks.to(self.config.device)
            
            # Sample random timesteps
            timesteps = self.scheduler.sample_timesteps(masks.size(0), self.config.device)
            
            # Sample noise
            noise = torch.randn_like(one_hot_features, device=self.config.device) * self.config.noise_magnitude
            
            # Add noise to features
            
            noised_features = self.scheduler.add_noise(one_hot_features, masks, noise, timesteps)
            
            # Predict noise
            self.optimizer.zero_grad()
            predicted_noise = self.model(noised_features, timesteps)
            
            # Calculate loss only on masked regions
            loss: torch.Tensor  = (F.mse_loss(predicted_noise, noise, reduction='none').mean(dim=2) * masks).mean()
            
            # Apply SNR weighting
            if self.config.use_snr_weighting:
                weights = self.scheduler.get_loss_weights(timesteps)
                loss = (loss * weights).mean()
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(self.dataloaders["train"])}] "
                      f"Loss: {loss.item():.7f}")
        
        self.lr_scheduler.step()
        train_loss = total_loss / len(self.dataloaders["train"])
        

        
        return train_loss
    
    @torch.no_grad()
    def _validate(self, epoch):
        """Validation epoch"""
        self.model.eval()
        total_loss = 0.0
        
        for one_hot_features, masks in self.dataloaders["validation"]:
            one_hot_features = one_hot_features.to(self.config.device)
            masks = masks.to(self.config.device)
            
            # Sample random timesteps
            timesteps = self.scheduler.sample_timesteps(masks.size(0), self.config.device)
            
            # Sample noise
            noise = torch.randn_like(one_hot_features, device=self.config.device) * self.config.noise_magnitude
            
            # Add noise to features
            noised_features = self.scheduler.add_noise(one_hot_features, masks, noise, timesteps)
    
            predicted_noise = self.model(noised_features, timesteps)
            
            loss: torch.Tensor  = (F.mse_loss(predicted_noise, noise, reduction='none').mean(dim=2) * masks).mean()
            
            # Apply SNR weighting
            if self.config.use_snr_weighting:
                weights = self.scheduler.get_loss_weights(timesteps)
                loss = (loss * weights).mean()
                
            total_loss += loss.item()
        
        validation_loss = total_loss / len(self.dataloaders["validation"])
        print(f"Validation Epoch: {epoch} Loss: {validation_loss:.7f}")
        return validation_loss
    
    def _save_checkpoint(self, epoch, loss, filename):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(self.config.output_dir, filename))
    
    def _plot_losses(self, train_losses, val_losses):
        """Plot training and validation losses"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=train_losses,
            name='Train Loss',
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            y=val_losses,
            name='Validation Loss',
            mode='lines'
        ))
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        fig.write_html(os.path.join(self.config.output_dir, 'loss_curve.html'))
        fig.write_image(os.path.join(self.config.output_dir, 'loss_curve.png'))
        
        if self.clearml_task is not None:
            self.logger.report_plotly(
                title="Training and Validation losses",
                series="", 
                figure=fig
            )
    
    @torch.no_grad()
    def generate_test_samples(self, only_one: bool = False):
        """Generate samples from test dataloader
        """
        self.model.eval()
        
        raw_samples = []
        reconstructed_samples = []
        
        for i, (one_hot_features, masks) in enumerate(self.dataloaders["test"]):
            # Move to device
            one_hot_features: torch.Tensor = one_hot_features.to(self.config.device)
            masks: torch.Tensor = masks.to(self.config.device)
            
            batch_size = one_hot_features.shape[0]
            
            # Start with noise in masked regions, original features elsewhere
            noise = torch.randn_like(one_hot_features, device=self.config.device) * self.config.noise_magnitude
            
            noised_features = (
                one_hot_features * (1 - masks.unsqueeze(-1)) + 
                noise * masks.unsqueeze(-1)
            )
            
            # Reverse diffusion process
            for t in range(self.scheduler.num_timesteps - 1, -1, -1):
                timesteps = torch.full((batch_size,), t, device=self.config.device)
                predicted_noise = self.model(noised_features, timesteps)
            
                # Update only masked regions
                noised_features = (
                    one_hot_features * (1 - masks.unsqueeze(-1)) + 
                    self.scheduler.step(
                        noised_features,
                        masks,
                        predicted_noise,
                        timesteps
                    ) * masks.unsqueeze(-1)
                )
                
            noised_features = torch.clamp(noised_features, 0, 1)
            
            noised_features_cpu = noised_features.cpu()
            raw_samples.append(noised_features_cpu)
            
            reconstructed = FishingAISDataset.one_hot_to_continuous(noised_features, masks, self.config.onehot_sizes)
            reconstructed_cpu = reconstructed.cpu()
            reconstructed_samples.append(reconstructed_cpu)
                
            # if i == 0:
            for j in range(batch_size):
                self.compare_heatmaps_plotly(one_hot_features, noised_features, masks, sample_idx=j, filename = "test_sample")
                break
            
            if only_one:
                break
            
        raw_samples_np = np.concatenate(raw_samples, axis=0)
        reconstructed_samples_np = np.concatenate(reconstructed_samples, axis=0)
        
        with open(f"{self.config.output_dir}/raw_samples.pkl", 'wb') as file:
            pickle.dump(raw_samples_np, file)
        with open(f"{self.config.output_dir}/reconstructed_samples.pkl", 'wb') as file:
            pickle.dump(reconstructed_samples_np, file)
                
        return raw_samples_np, reconstructed_samples_np  
    
    def compare_heatmaps_plotly(self, original, noised, mask, sample_idx: int = 0, filename: str = "noizing_sample", add_red: bool = True):
        """
        Args:
            original (torch.Tensor): [2, 16, 622] (one_hot_features)
            noised (torch.Tensor): [2, 16, 622] (noised_features)
            mask (torch.Tensor): [2, 16] (masks)
            sample_idx (int): Which batch sample to visualize (0 or 1)
        """
        # Convert to numpy and take argmax (if one-hot) or direct values
        original_np = original[sample_idx].cpu().numpy()  # [16, 622]
        noised_np: np.ndarray = noised[sample_idx].cpu().numpy()      # [16, 622]
        mask_np = mask[sample_idx].cpu().numpy()          # [16]

        noised_np = noised_np.__abs__()
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Original (Sample {sample_idx})",
                "Noised (Masked Regions can be in Red)"
            ),
            horizontal_spacing=0.05
        )

        # Original heatmap
        fig.add_trace(
            go.Heatmap(
                z=original_np,
                colorscale=[[0, "black"], [1, "white"]],
                colorbar=dict(title="Value"),
                hoverinfo="x+y+z",
                name="Original"
            ),
            row=1, col=1
        )

        # Noised heatmap (with masked regions highlighted)
        fig.add_trace(
            go.Heatmap(
                z=noised_np,
                colorscale=[[0, "black"], [1, "white"]],
                colorbar=dict(title="Value"),
                hoverinfo="x+y+z",
                name="Noised"
            ),
            row=1, col=2
        )

        # Overlay red rectangles where mask=1
        if add_red:
            for pos in np.where(mask_np == 1)[0]:
                fig.add_shape(
                    type="rect",
                    x0=-0.5, x1=sum(self.config.onehot_sizes) - 0.5,  # Cover full width (622 columns)
                    y0=pos-0.5, y1=pos+0.5,
                    fillcolor="rgba(255,0,0,0.2)",
                    row=1, col=2
                )

        # Update layout
        fig.update_layout(
            height=1000,
            width=2000,
            title_text=f"16×622 Features Comparison (Batch Sample {sample_idx})",
            xaxis_title="Feature Dimension",
            yaxis_title="Sequence Position (16)",
        )

        fig.write_html(os.path.join(self.config.output_dir, f'{filename}_{sample_idx}.html'))
        fig.write_image(os.path.join(self.config.output_dir, f'{filename}_{sample_idx}.png'))
        if self.clearml_task is not None:
            self.logger.report_plotly(
                title="Test vs Prediction Comparison", 
                series=f"Sample_{sample_idx}", 
                figure=fig
            )
    
"""
CTGAN (Conditional Tabular GAN) Model for Data Augmentation
Handles class imbalance by generating high-quality synthetic minority samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple, Dict
import os


class ConditionalGenerator(nn.Module):
    """Generator network for CTGAN"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, 
                 n_layers: int = 3, condition_dim: int = 1):
        """
        Args:
            input_dim: dimension of noise vector
            output_dim: dimension of generated data
            hidden_dim: dimension of hidden layers
            n_layers: number of hidden layers
            condition_dim: dimension of condition vector (e.g., class label)
        """
        super(ConditionalGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        
        # Build network
        layers = []
        in_dim = input_dim + condition_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: noise vector [batch_size, input_dim]
            condition: condition vector [batch_size, condition_dim]
        
        Returns:
            generated data [batch_size, output_dim]
        """
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        
        # Generate data
        output = self.network(x)
        
        return output


class ConditionalDiscriminator(nn.Module):
    """Discriminator network for CTGAN"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 n_layers: int = 3, condition_dim: int = 1):
        """
        Args:
            input_dim: dimension of input data
            hidden_dim: dimension of hidden layers
            n_layers: number of hidden layers
            condition_dim: dimension of condition vector (e.g., class label)
        """
        super(ConditionalDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # Build network
        layers = []
        in_dim = input_dim + condition_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, data: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: real or generated data [batch_size, input_dim]
            condition: condition vector [batch_size, condition_dim]
        
        Returns:
            probability of being real [batch_size, 1]
        """
        # Concatenate data and condition
        x = torch.cat([data, condition], dim=1)
        
        # Discriminate
        output = self.network(x)
        
        return output


class TabularDataset(Dataset):
    """Dataset for tabular data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: feature array [n_samples, n_features]
            labels: label array [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CTGAN:
    """Conditional Tabular GAN for data augmentation"""
    
    def __init__(self, n_features: int, n_classes: int = 2, 
                 noise_dim: int = 100, hidden_dim: int = 256, 
                 n_layers: int = 3, lr: float = 0.0002, 
                 beta1: float = 0.5, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            n_features: number of features
            n_classes: number of classes
            noise_dim: dimension of noise vector
            hidden_dim: dimension of hidden layers
            n_layers: number of hidden layers
            lr: learning rate
            beta1: beta1 for Adam optimizer
            device: device to use
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.noise_dim = noise_dim
        self.device = device
        
        # Initialize generator and discriminator
        self.generator = ConditionalGenerator(
            input_dim=noise_dim, 
            output_dim=n_features, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers,
            condition_dim=1
        ).to(device)
        
        self.discriminator = ConditionalDiscriminator(
            input_dim=n_features, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers,
            condition_dim=1
        ).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """
        Train for one epoch
        
        Args:
            dataloader: training data loader
            epoch: current epoch number
        """
        self.generator.train()
        self.discriminator.train()
        
        g_loss_epoch = 0
        d_loss_epoch = 0
        
        for i, (real_data, real_labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            # Move to device
            real_data = real_data.to(self.device)
            real_labels = real_labels.to(self.device).unsqueeze(1)
            
            # Labels for real and fake data
            real_labels_tensor = torch.ones(batch_size, 1).to(self.device)
            fake_labels_tensor = torch.zeros(batch_size, 1).to(self.device)
            
            # ================== Train Discriminator ==================
            self.d_optimizer.zero_grad()
            
            # Real data
            d_real = self.discriminator(real_data, real_labels)
            d_loss_real = self.criterion(d_real, real_labels_tensor)
            
            # Fake data
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise, real_labels)
            d_fake = self.discriminator(fake_data.detach(), real_labels)
            d_loss_fake = self.criterion(d_fake, fake_labels_tensor)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()
            
            # ================== Train Generator ==================
            self.g_optimizer.zero_grad()
            
            # Generate fake data
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise, real_labels)
            
            # Discriminator output for fake data
            d_fake = self.discriminator(fake_data, real_labels)
            
            # Generator loss (want discriminator to think fake is real)
            g_loss = self.criterion(d_fake, real_labels_tensor)
            g_loss.backward()
            self.g_optimizer.step()
            
            # Record losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
        
        avg_g_loss = g_loss_epoch / len(dataloader)
        avg_d_loss = d_loss_epoch / len(dataloader)
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        
        return avg_g_loss, avg_d_loss
    
    def generate_samples(self, n_samples: int, class_label: int) -> np.ndarray:
        """
        Generate synthetic samples for a specific class
        
        Args:
            n_samples: number of samples to generate
            class_label: class label for conditional generation
        
        Returns:
            generated samples [n_samples, n_features]
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim).to(self.device)
            condition = torch.full((n_samples, 1), class_label).to(self.device)
            
            generated = self.generator(noise, condition)
            
        return generated.cpu().numpy()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 100, batch_size: int = 64, 
            minority_class: int = 1, verbose: bool = True):
        """
        Train the CTGAN model
        
        Args:
            X_train: training features
            y_train: training labels
            epochs: number of training epochs
            batch_size: batch size
            minority_class: minority class label
            verbose: whether to print training progress
        """
        # Create dataset and dataloader
        dataset = TabularDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose:
            print(f"Training CTGAN for {epochs} epochs...")
            print(f"Minority class: {minority_class}")
            print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            g_loss, d_loss = self.train_epoch(dataloader, epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
    
    def augment_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                    n_samples: int = None, minority_class: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data by generating synthetic minority samples
        
        Args:
            X_train: training features
            y_train: training labels
            n_samples: number of samples to generate (default: balance classes)
            minority_class: minority class label
        
        Returns:
            augmented features and labels
        """
        # Calculate number of samples to generate
        if n_samples is None:
            majority_count = np.sum(y_train != minority_class)
            minority_count = np.sum(y_train == minority_class)
            n_samples = majority_count - minority_count
        
        # Generate synthetic samples
        synthetic_samples = self.generate_samples(n_samples, minority_class)
        synthetic_labels = np.full(n_samples, minority_class)
        
        # Combine with original data
        X_augmented = np.vstack([X_train, synthetic_samples])
        y_augmented = np.concatenate([y_train, synthetic_labels])
        
        print(f"Generated {n_samples} synthetic samples for class {minority_class}")
        print(f"Original dataset: {len(X_train)} samples")
        print(f"Augmented dataset: {len(X_augmented)} samples")
        print(f"New class distribution: {np.bincount(y_augmented.astype(int))}")
        
        return X_augmented, y_augmented
    
    def save_model(self, save_dir: str = 'models'):
        """Save the trained model"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'noise_dim': self.noise_dim
        }, os.path.join(save_dir, 'ctgan_model.pth'))
        
        print(f"Model saved to {save_dir}/ctgan_model.pth")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        
        print(f"Model loaded from {model_path}")


if __name__ == '__main__':
    # Example usage
    print("CTGAN Model for Data Augmentation")
    print("=" * 50)
    
    # Generate synthetic imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.zeros(n_samples)
    y_train[:100] = 1  # 10% minority class
    
    print(f"Original class distribution: {np.bincount(y_train.astype(int))}")
    
    # Initialize CTGAN
    ctgan = CTGAN(n_features=n_features, n_classes=2, noise_dim=100, hidden_dim=256)
    
    # Train CTGAN
    ctgan.fit(X_train, y_train, epochs=50, batch_size=64, minority_class=1)
    
    # Augment data
    X_augmented, y_augmented = ctgan.augment_data(X_train, y_train, minority_class=1)
    
    # Save model
    ctgan.save_model()
    
    print("\nCTGAN training and augmentation completed!")

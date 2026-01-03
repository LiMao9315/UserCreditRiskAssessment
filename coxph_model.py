"""
SHAP-Enhanced Cox Proportional Hazards Model for Survival Analysis
Integrates temporal survival data with graph embeddings for interpretable risk predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import os
from scipy.stats import rankdata


class CoxPHModel(nn.Module):
    """Cox Proportional Hazards Model"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        """
        Args:
            input_dim: dimension of input features
            hidden_dims: list of hidden layer dimensions (None for linear model)
        """
        super(CoxPHModel, self).__init__()
        
        self.input_dim = input_dim
        self.use_nn = hidden_dims is not None
        
        if self.use_nn:
            # Neural network version
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.network = nn.Sequential(*layers)
        else:
            # Linear version (standard Cox model)
            self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features [batch_size, input_dim]
        
        Returns:
            risk scores [batch_size, 1]
        """
        if self.use_nn:
            return self.network(x)
        else:
            return self.linear(x)
    
    def get_risk_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Get risk scores for input data
        
        Args:
            x: input features [n_samples, input_dim]
        
        Returns:
            risk scores [n_samples]
        """
        self.eval()
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            risk_scores = self.forward(x_tensor)
            
        return risk_scores.numpy().flatten()


class CoxPHLoss(nn.Module):
    """Negative partial log-likelihood loss for Cox model"""
    
    def __init__(self):
        super(CoxPHLoss, self).__init__()
    
    def forward(self, risk_scores: torch.Tensor, 
                 survival_time: torch.Tensor, 
                 event: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_scores: predicted risk scores [batch_size, 1]
            survival_time: survival times [batch_size]
            event: event indicators [batch_size] (1 if event occurred, 0 if censored)
        
        Returns:
            negative partial log-likelihood loss
        """
        # Sort by survival time (descending)
        indices = torch.argsort(survival_time, descending=True)
        risk_scores = risk_scores[indices]
        event = event[indices]
        
        # Compute log-sum-exp for each at-risk set
        log_risk_sum = torch.logcumsumexp(risk_scores, dim=0)
        
        # Compute partial log-likelihood
        log_likelihood = torch.sum(event * (risk_scores - log_risk_sum))
        
        # Return negative log-likelihood
        return -log_likelihood


class SHAPExplainer:
    """SHAP explainer for Cox model"""
    
    def __init__(self, model: CoxPHModel, background_data: np.ndarray, 
                 n_samples: int = 100):
        """
        Args:
            model: trained Cox model
            background_data: background data for SHAP values
            n_samples: number of samples for SHAP approximation
        """
        self.model = model
        self.background_data = background_data
        self.n_samples = n_samples
        self.device = next(model.parameters()).device
        
    def explain(self, x: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for input data
        
        Args:
            x: input data [n_samples, n_features]
        
        Returns:
            SHAP values [n_samples, n_features]
        """
        n_samples = x.shape[0]
        n_features = x.shape[1]
        
        # Use Kernel SHAP approximation
        shap_values = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            shap_values[i] = self._compute_shap_single(x[i])
        
        return shap_values
    
    def _compute_shap_single(self, x_single: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a single sample
        
        Args:
            x_single: single sample [n_features]
        
        Returns:
            SHAP values [n_features]
        """
        n_features = x_single.shape[0]
        shap_values = np.zeros(n_features)
        
        # Expected value (average prediction on background data)
        expected_value = self._predict_batch(self.background_data).mean()
        
        # Individual prediction
        individual_value = self._predict_single(x_single)
        
        # Compute SHAP values using sampling approximation
        for j in range(n_features):
            # Create coalition with feature j
            x_with_j = self.background_data.copy()
            x_with_j[:, j] = x_single[j]
            
            x_without_j = self.background_data.copy()
            x_without_j[:, j] = self.background_data[:, j]
            
            # Compute marginal contribution
            pred_with_j = self._predict_batch(x_with_j).mean()
            pred_without_j = self._predict_batch(x_without_j).mean()
            
            shap_values[j] = pred_with_j - pred_without_j
        
        # Ensure SHAP values sum to the difference
        shap_sum = shap_values.sum()
        if shap_sum != 0:
            shap_values = shap_values * (individual_value - expected_value) / shap_sum
        
        return shap_values
    
    def _predict_single(self, x: np.ndarray) -> float:
        """Predict risk score for single sample"""
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        risk_score = self.model(x_tensor)
        return risk_score.item()
    
    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        """Predict risk scores for batch"""
        x_tensor = torch.FloatTensor(x).to(self.device)
        risk_scores = self.model(x_tensor)
        return risk_scores.cpu().numpy().flatten()


class CoxPHTrainer:
    """Trainer for Cox PH model"""
    
    def __init__(self, model: CoxPHModel, lr: float = 0.001, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: Cox PH model
            lr: learning rate
            device: device to use
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = CoxPHLoss()
        self.losses = []
        
    def train_epoch(self, features: torch.Tensor, 
                    survival_time: torch.Tensor, 
                    event: torch.Tensor) -> float:
        """
        Train for one epoch
        
        Args:
            features: input features
            survival_time: survival times
            event: event indicators
        
        Returns:
            loss value
        """
        self.model.train()
        
        # Forward pass
        risk_scores = self.model(features)
        
        # Compute loss
        loss = self.criterion(risk_scores, survival_time, event)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        return loss.item()
    
    def fit(self, features: np.ndarray, survival_time: np.ndarray, 
            event: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, verbose: bool = True):
        """
        Train the Cox PH model
        
        Args:
            features: input features
            survival_time: survival times
            event: event indicators
            epochs: number of training epochs
            batch_size: batch size
            verbose: whether to print training progress
        """
        n_samples = len(features)
        
        # Convert to tensors
        features = torch.FloatTensor(features).to(self.device)
        survival_time = torch.FloatTensor(survival_time).to(self.device)
        event = torch.FloatTensor(event).to(self.device)
        
        if verbose:
            print(f"Training Cox PH model for {epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Samples: {n_samples}")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_features = features[batch_indices]
                batch_time = survival_time[batch_indices]
                batch_event = event[batch_indices]
                
                loss = self.train_epoch(batch_features, batch_time, batch_event)
                
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    
    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        """
        Predict risk scores
        
        Args:
            features: input features
        
        Returns:
            risk scores
        """
        return self.model.get_risk_scores(features)
    
    def get_shap_explanations(self, features: np.ndarray, 
                             background_data: np.ndarray = None,
                             n_samples: int = 100) -> np.ndarray:
        """
        Get SHAP explanations
        
        Args:
            features: input features to explain
            background_data: background data for SHAP
            n_samples: number of samples for SHAP approximation
        
        Returns:
            SHAP values
        """
        if background_data is None:
            background_data = features
        
        explainer = SHAPExplainer(self.model, background_data, n_samples)
        shap_values = explainer.explain(features)
        
        return shap_values
    
    def save_model(self, save_dir: str = 'models'):
        """Save the trained model"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses
        }, os.path.join(save_dir, 'coxph_model.pth'))
        
        print(f"Model saved to {save_dir}/coxph_model.pth")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']
        
        print(f"Model loaded from {model_path}")


def concordance_index(risk_scores: np.ndarray, 
                     survival_time: np.ndarray, 
                     event: np.ndarray) -> float:
    """
    Compute Concordance Index (C-index)
    
    Args:
        risk_scores: predicted risk scores
        survival_time: actual survival times
        event: event indicators
    
    Returns:
        C-index
    """
    # Convert to ranks
    risk_ranks = rankdata(-risk_scores)  # Higher risk = lower rank
    
    # Count comparable pairs
    n_comparable = 0
    n_concordant = 0
    
    for i in range(len(risk_scores)):
        for j in range(i + 1, len(risk_scores)):
            # Check if pair is comparable
            if event[i] == 1 and survival_time[i] < survival_time[j]:
                n_comparable += 1
                if risk_ranks[i] > risk_ranks[j]:
                    n_concordant += 1
            elif event[j] == 1 and survival_time[j] < survival_time[i]:
                n_comparable += 1
                if risk_ranks[j] > risk_ranks[i]:
                    n_concordant += 1
    
    # Compute C-index
    if n_comparable == 0:
        return 0.5
    
    return n_concordant / n_comparable


if __name__ == '__main__':
    # Example usage
    print("SHAP-Enhanced Cox PH Model for Survival Analysis")
    print("=" * 50)
    
    # Generate synthetic survival data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    features = np.random.randn(n_samples, n_features)
    survival_time = np.random.exponential(scale=36, size=n_samples)
    event = np.random.binomial(1, 0.3, size=n_samples)
    
    print(f"Data: {n_samples} samples, {n_features} features")
    print(f"Events: {event.sum()} ({event.mean()*100:.1f}%)")
    
    # Initialize Cox PH model
    model = CoxPHModel(input_dim=n_features, hidden_dims=[64, 32])
    
    # Train model
    trainer = CoxPHTrainer(model, lr=0.001)
    trainer.fit(features, survival_time, event, epochs=50)
    
    # Predict risk scores
    risk_scores = trainer.predict_risk(features)
    
    # Compute C-index
    c_index = concordance_index(risk_scores, survival_time, event)
    print(f"\nC-index: {c_index:.4f}")
    
    # Get SHAP explanations
    shap_values = trainer.get_shap_explanations(features[:100])
    print(f"SHAP values shape: {shap_values.shape}")
    
    # Save model
    trainer.save_model()
    
    print("\nCox PH model training completed!")

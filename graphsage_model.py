"""
GraphSAGE (Graph Sample and Aggregate) Model for Graph Embedding
Learns node embeddings that capture heterogeneous risk dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import os


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer"""
    
    def __init__(self, in_features: int, out_features: int, 
                 aggregator: str = 'mean', dropout: float = 0.5):
        """
        Args:
            in_features: input feature dimension
            out_features: output feature dimension
            aggregator: aggregation method ('mean', 'max', 'sum')
            dropout: dropout rate
        """
        super(GraphSAGELayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # Linear transformation for self features
        self.weight_self = nn.Linear(in_features, out_features)
        
        # Linear transformation for neighbor features
        self.weight_neighbor = nn.Linear(in_features, out_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features [n_nodes, in_features]
            adj: adjacency matrix [n_nodes, n_nodes]
        
        Returns:
            updated node features [n_nodes, out_features]
        """
        # Self transformation
        self_features = self.weight_self(x)
        
        # Neighbor aggregation
        if self.aggregator == 'mean':
            # Mean aggregation
            neighbor_features = torch.matmul(adj, x) / (adj.sum(dim=1, keepdim=True) + 1e-8)
        elif self.aggregator == 'sum':
            # Sum aggregation
            neighbor_features = torch.matmul(adj, x)
        elif self.aggregator == 'max':
            # Max aggregation
            neighbor_features = torch.zeros_like(x)
            for i in range(x.size(0)):
                neighbors = adj[i] > 0
                if neighbors.sum() > 0:
                    neighbor_features[i] = x[neighbors].max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        neighbor_features = self.weight_neighbor(neighbor_features)
        
        # Combine self and neighbor features
        output = self_features + neighbor_features
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class GraphSAGE(nn.Module):
    """GraphSAGE model for node embedding"""
    
    def __init__(self, in_features: int, hidden_dims: List[int], 
                 out_features: int, aggregator: str = 'mean', 
                 dropout: float = 0.5):
        """
        Args:
            in_features: input feature dimension
            hidden_dims: list of hidden layer dimensions
            out_features: output embedding dimension
            aggregator: aggregation method ('mean', 'max', 'sum')
            dropout: dropout rate
        """
        super(GraphSAGE, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Build layers
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.append(GraphSAGELayer(prev_dim, hidden_dim, aggregator, dropout))
            prev_dim = hidden_dim
        
        layers.append(GraphSAGELayer(prev_dim, out_features, aggregator, dropout))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features [n_nodes, in_features]
            adj: adjacency matrix [n_nodes, n_nodes]
        
        Returns:
            node embeddings [n_nodes, out_features]
        """
        # Pass through layers
        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
        
        return x


class CreditGraphBuilder:
    """Build borrower relational graphs"""
    
    def __init__(self, n_neighbors: int = 5, similarity_threshold: float = 0.7):
        """
        Args:
            n_neighbors: number of neighbors for k-NN graph
            similarity_threshold: threshold for edge creation
        """
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        
    def build_knn_graph(self, features: np.ndarray) -> torch.Tensor:
        """
        Build k-NN graph based on feature similarity
        
        Args:
            features: node features [n_nodes, n_features]
        
        Returns:
            adjacency matrix [n_nodes, n_nodes]
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_nodes = features.shape[0]
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto')
        nbrs.fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        # Build adjacency matrix
        adj = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j, idx in enumerate(indices[i]):
                if j > 0:  # Skip self
                    adj[i, idx] = 1.0
                    adj[idx, i] = 1.0  # Symmetric
        
        return torch.FloatTensor(adj)
    
    def build_similarity_graph(self, features: np.ndarray, 
                               metric: str = 'cosine') -> torch.Tensor:
        """
        Build similarity graph based on pairwise similarity
        
        Args:
            features: node features [n_nodes, n_features]
            metric: similarity metric ('cosine', 'euclidean')
        
        Returns:
            adjacency matrix [n_nodes, n_nodes]
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        n_nodes = features.shape[0]
        
        # Compute similarity matrix
        if metric == 'cosine':
            similarity = cosine_similarity(features)
        elif metric == 'euclidean':
            distances = euclidean_distances(features)
            # Convert distances to similarities
            max_dist = distances.max()
            similarity = 1 - (distances / max_dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Apply threshold
        adj = (similarity >= self.similarity_threshold).astype(float)
        
        # Remove self-loops
        np.fill_diagonal(adj, 0)
        
        return torch.FloatTensor(adj)
    
    def build_domain_graph(self, features: np.ndarray, 
                          categorical_features: Dict[str, List[int]]) -> torch.Tensor:
        """
        Build graph based on domain knowledge (e.g., same category)
        
        Args:
            features: node features [n_nodes, n_features]
            categorical_features: dictionary of feature name to indices
        
        Returns:
            adjacency matrix [n_nodes, n_nodes]
        """
        n_nodes = features.shape[0]
        adj = np.zeros((n_nodes, n_nodes))
        
        # Connect nodes with similar categorical features
        for feat_name, feat_indices in categorical_features.items():
            for idx in feat_indices:
                # Find nodes with same value
                values = features[:, idx]
                for i in range(n_nodes):
                    for j in range(i + 1, n_nodes):
                        if values[i] == values[j]:
                            adj[i, j] += 1
                            adj[j, i] += 1
        
        # Normalize
        adj = adj / adj.max()
        
        return torch.FloatTensor(adj)


class GraphSAGETrainer:
    """Trainer for GraphSAGE model"""
    
    def __init__(self, model: GraphSAGE, lr: float = 0.001, 
                 weight_decay: float = 5e-4, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: GraphSAGE model
            lr: learning rate
            weight_decay: weight decay for regularization
            device: device to use
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.losses = []
        
    def train_epoch(self, features: torch.Tensor, adj: torch.Tensor, 
                   labels: torch.Tensor = None, mask: torch.Tensor = None):
        """
        Train for one epoch
        
        Args:
            features: node features
            adj: adjacency matrix
            labels: node labels (optional)
            mask: training mask (optional)
        """
        self.model.train()
        
        # Forward pass
        embeddings = self.model(features, adj)
        
        # Compute loss
        if labels is not None:
            if mask is not None:
                # Supervised learning with mask
                loss = F.cross_entropy(embeddings[mask], labels[mask])
            else:
                # Supervised learning
                loss = F.cross_entropy(embeddings, labels)
        else:
            # Unsupervised learning (reconstruction loss)
            loss = F.mse_loss(embeddings, features[:, :embeddings.size(1)])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        return loss.item()
    
    def fit(self, features: np.ndarray, adj: np.ndarray, 
            labels: np.ndarray = None, mask: np.ndarray = None,
            epochs: int = 100, verbose: bool = True):
        """
        Train the GraphSAGE model
        
        Args:
            features: node features
            adj: adjacency matrix
            labels: node labels (optional)
            mask: training mask (optional)
            epochs: number of training epochs
            verbose: whether to print training progress
        """
        # Convert to tensors
        features = torch.FloatTensor(features).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
        
        if labels is not None:
            labels = torch.LongTensor(labels).to(self.device)
        if mask is not None:
            mask = torch.BoolTensor(mask).to(self.device)
        
        if verbose:
            print(f"Training GraphSAGE for {epochs} epochs...")
            print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            loss = self.train_epoch(features, adj, labels, mask)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f}")
    
    def get_embeddings(self, features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Get node embeddings
        
        Args:
            features: node features
            adj: adjacency matrix
        
        Returns:
            node embeddings
        """
        self.model.eval()
        
        with torch.no_grad():
            features = torch.FloatTensor(features).to(self.device)
            adj = torch.FloatTensor(adj).to(self.device)
            
            embeddings = self.model(features, adj)
            
        return embeddings.cpu().numpy()
    
    def save_model(self, save_dir: str = 'models'):
        """Save the trained model"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses
        }, os.path.join(save_dir, 'graphsage_model.pth'))
        
        print(f"Model saved to {save_dir}/graphsage_model.pth")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']
        
        print(f"Model loaded from {model_path}")


if __name__ == '__main__':
    # Example usage
    print("GraphSAGE Model for Graph Embedding")
    print("=" * 50)
    
    # Generate synthetic graph data
    np.random.seed(42)
    n_nodes = 100
    n_features = 20
    
    features = np.random.randn(n_nodes, n_features)
    
    # Build graph
    graph_builder = CreditGraphBuilder(n_neighbors=5)
    adj = graph_builder.build_knn_graph(features)
    
    print(f"Graph built: {n_nodes} nodes, {adj.sum().int().item() / 2} edges")
    
    # Initialize GraphSAGE
    model = GraphSAGE(
        in_features=n_features,
        hidden_dims=[64, 32],
        out_features=16,
        aggregator='mean',
        dropout=0.5
    )
    
    # Train GraphSAGE
    trainer = GraphSAGETrainer(model, lr=0.001)
    trainer.fit(features, adj, epochs=50)
    
    # Get embeddings
    embeddings = trainer.get_embeddings(features, adj)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Save model
    trainer.save_model()
    
    print("\nGraphSAGE training completed!")

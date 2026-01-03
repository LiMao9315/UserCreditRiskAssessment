"""
Model Training Script for Credit Risk Assessment
Integrates CTGAN, GraphSAGE, and Cox PH models for comprehensive training
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple
import argparse
import json
from datetime import datetime

from data_preprocessing import DataPreprocessor, CreditRiskDataset
from ctgan_model import CTGAN
from graphsage_model import GraphSAGE, CreditGraphBuilder, GraphSAGETrainer
from coxph_model import CoxPHModel, CoxPHTrainer


class ModelTrainingPipeline:
    """Complete training pipeline for credit risk assessment models"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            data_path=config['data_path'],
            target_column=config['target_column']
        )
        
        # Store trained models
        self.models = {}
        self.training_history = {}
        
        print(f"Training pipeline initialized on device: {self.device}")
    
    def load_and_preprocess_data(self) -> Tuple:
        """
        Load and preprocess data
        
        Returns:
            train_data, test_data, feature_names
        """
        print("\n" + "="*50)
        print("Step 1: Loading and Preprocessing Data")
        print("="*50)
        
        # Load data
        data = self.preprocessor.load_data()
        print(f"Data loaded: {data.shape}")
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_data(data)
        print(f"Data preprocessed: {processed_data.shape}")
        
        # Split data
        train_data, test_data = self.preprocessor.split_data(
            processed_data, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        print(f"Train data: {train_data.shape}")
        print(f"Test data: {test_data.shape}")
        
        # Get feature names
        feature_names = self.preprocessor.get_feature_names()
        print(f"Features: {len(feature_names)}")
        
        return train_data, test_data, feature_names
    
    def train_ctgan(self, train_data: pd.DataFrame) -> CTGAN:
        """
        Train CTGAN for data augmentation
        
        Args:
            train_data: training data
        
        Returns:
            trained CTGAN model
        """
        print("\n" + "="*50)
        print("Step 2: Training CTGAN for Data Augmentation")
        print("="*50)
        
        # Initialize CTGAN
        ctgan = CTGAN(
            embedding_dim=self.config['ctgan']['embedding_dim'],
            generator_dim=self.config['ctgan']['generator_dim'],
            discriminator_dim=self.config['ctgan']['discriminator_dim'],
            generator_lr=self.config['ctgan']['generator_lr'],
            discriminator_lr=self.config['ctgan']['discriminator_lr'],
            batch_size=self.config['ctgan']['batch_size'],
            epochs=self.config['ctgan']['epochs'],
            device=self.device
        )
        
        # Train CTGAN
        ctgan.fit(train_data, discrete_columns=self.config['discrete_columns'])
        
        # Save model
        ctgan.save_model(os.path.join(self.config['model_dir'], 'ctgan_model.pth'))
        
        # Generate synthetic data
        n_samples = int(len(train_data) * self.config['augmentation_ratio'])
        synthetic_data = ctgan.generate_samples(n_samples)
        
        # Save synthetic data
        synthetic_path = os.path.join(self.config['output_dir'], 'synthetic_data.csv')
        synthetic_data.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to {synthetic_path}")
        
        self.models['ctgan'] = ctgan
        self.training_history['ctgan'] = {
            'losses': ctgan.losses,
            'n_synthetic_samples': n_samples
        }
        
        return ctgan
    
    def train_graphsage(self, train_data: pd.DataFrame, 
                       feature_names: list) -> GraphSAGE:
        """
        Train GraphSAGE for graph embeddings
        
        Args:
            train_data: training data
            feature_names: feature names
        
        Returns:
            trained GraphSAGE model
        """
        print("\n" + "="*50)
        print("Step 3: Training GraphSAGE for Graph Embeddings")
        print("="*50)
        
        # Prepare features
        features = train_data[feature_names].values
        labels = train_data[self.config['target_column']].values
        
        # Build graph
        graph_builder = CreditGraphBuilder(
            graph_type=self.config['graphsage']['graph_type'],
            k_neighbors=self.config['graphsage']['k_neighbors']
        )
        
        edge_index = graph_builder.build_graph(features, labels)
        print(f"Graph built: {edge_index.shape[1]} edges")
        
        # Initialize GraphSAGE
        input_dim = features.shape[1]
        hidden_dim = self.config['graphsage']['hidden_dim']
        output_dim = self.config['graphsage']['output_dim']
        
        graphsage = GraphSAGE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=self.config['graphsage']['num_layers'],
            dropout=self.config['graphsage']['dropout']
        ).to(self.device)
        
        # Train GraphSAGE
        trainer = GraphSAGETrainer(
            model=graphsage,
            lr=self.config['graphsage']['learning_rate'],
            device=self.device
        )
        
        trainer.fit(
            edge_index=edge_index,
            node_features=features,
            node_labels=labels,
            epochs=self.config['graphsage']['epochs'],
            batch_size=self.config['graphsage']['batch_size']
        )
        
        # Save model
        trainer.save_model(os.path.join(self.config['model_dir'], 'graphsage_model.pth'))
        
        # Generate embeddings
        embeddings = trainer.get_embeddings(features)
        print(f"Embeddings generated: {embeddings.shape}")
        
        # Save embeddings
        embeddings_path = os.path.join(self.config['output_dir'], 'train_embeddings.npy')
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to {embeddings_path}")
        
        self.models['graphsage'] = graphsage
        self.training_history['graphsage'] = {
            'losses': trainer.losses,
            'embeddings_shape': embeddings.shape
        }
        
        return graphsage
    
    def train_coxph(self, train_data: pd.DataFrame, 
                   feature_names: list) -> CoxPHModel:
        """
        Train Cox PH model for survival analysis
        
        Args:
            train_data: training data
            feature_names: feature names
        
        Returns:
            trained Cox PH model
        """
        print("\n" + "="*50)
        print("Step 4: Training Cox PH Model for Survival Analysis")
        print("="*50)
        
        # Prepare features
        features = train_data[feature_names].values
        
        # Generate synthetic survival data (if not present)
        if 'survival_time' not in train_data.columns:
            # Simulate survival time based on credit risk
            np.random.seed(self.config['random_state'])
            risk_scores = train_data[self.config['target_column']].values
            survival_time = np.random.exponential(
                scale=36 / (1 + risk_scores),  # Higher risk = shorter survival
                size=len(train_data)
            )
            event = np.random.binomial(1, 0.3, size=len(train_data))
        else:
            survival_time = train_data['survival_time'].values
            event = train_data['event'].values
        
        print(f"Survival data: {event.sum()} events ({event.mean()*100:.1f}%)")
        
        # Initialize Cox PH model
        input_dim = features.shape[1]
        hidden_dims = self.config['coxph']['hidden_dims']
        
        coxph = CoxPHModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims
        )
        
        # Train Cox PH model
        trainer = CoxPHTrainer(
            model=coxph,
            lr=self.config['coxph']['learning_rate'],
            device=self.device
        )
        
        trainer.fit(
            features=features,
            survival_time=survival_time,
            event=event,
            epochs=self.config['coxph']['epochs'],
            batch_size=self.config['coxph']['batch_size']
        )
        
        # Save model
        trainer.save_model(os.path.join(self.config['model_dir'], 'coxph_model.pth'))
        
        # Predict risk scores
        risk_scores = trainer.predict_risk(features)
        print(f"Risk scores generated: {risk_scores.shape}")
        
        # Save risk scores
        risk_scores_path = os.path.join(self.config['output_dir'], 'train_risk_scores.npy')
        np.save(risk_scores_path, risk_scores)
        print(f"Risk scores saved to {risk_scores_path}")
        
        self.models['coxph'] = coxph
        self.training_history['coxph'] = {
            'losses': trainer.losses,
            'risk_scores_shape': risk_scores.shape
        }
        
        return coxph
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "="*50)
        print("Starting Full Training Pipeline")
        print("="*50)
        
        start_time = datetime.now()
        
        # Step 1: Load and preprocess data
        train_data, test_data, feature_names = self.load_and_preprocess_data()
        
        # Step 2: Train CTGAN
        self.train_ctgan(train_data)
        
        # Step 3: Train GraphSAGE
        self.train_graphsage(train_data, feature_names)
        
        # Step 4: Train Cox PH
        self.train_coxph(train_data, feature_names)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*50)
        print("Training Pipeline Completed!")
        print("="*50)
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Models saved to: {self.config['model_dir']}")
        print(f"Outputs saved to: {self.config['output_dir']}")
        
        # Save training history
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        return self.models


def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'data_path': 'credit_risk_data.csv',
        'target_column': 'default',
        'test_size': 0.2,
        'random_state': 42,
        'augmentation_ratio': 0.5,
        'discrete_columns': ['education', 'marriage'],
        'output_dir': 'outputs',
        'model_dir': 'models',
        'log_dir': 'logs',
        'ctgan': {
            'embedding_dim': 128,
            'generator_dim': [256, 256],
            'discriminator_dim': [256, 256],
            'generator_lr': 2e-4,
            'discriminator_lr': 2e-4,
            'batch_size': 500,
            'epochs': 300
        },
        'graphsage': {
            'graph_type': 'knn',
            'k_neighbors': 10,
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100
        },
        'coxph': {
            'hidden_dims': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train credit risk assessment models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='credit_risk_data.csv',
                       help='Path to data file')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='models',
                       help='Model directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Update config with command line arguments
    config['data_path'] = args.data
    config['output_dir'] = args.output
    config['model_dir'] = args.model
    
    # Run training pipeline
    pipeline = ModelTrainingPipeline(config)
    models = pipeline.run_full_pipeline()
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

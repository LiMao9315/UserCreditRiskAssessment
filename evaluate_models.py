"""
Model Validation and Evaluation Script for Credit Risk Assessment
Comprehensive evaluation of CTGAN, GraphSAGE, and Cox PH models
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy.stats import ks_2samp

from data_preprocessing import DataPreprocessor
from ctgan_model import CTGAN
from graphsage_model import GraphSAGE, GraphSAGETrainer
from coxph_model import CoxPHModel, CoxPHTrainer, concordance_index


class ModelEvaluator:
    """Comprehensive evaluator for credit risk models"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor(
            data_path=config['data_path'],
            target_column=config['target_column']
        )
        
        # Load trained models
        self.models = {}
        self.load_models()
        
        # Store evaluation results
        self.evaluation_results = {}
        
        print(f"Evaluator initialized on device: {self.device}")
    
    def load_models(self):
        """Load trained models"""
        model_dir = self.config['model_dir']
        
        # Load CTGAN
        if os.path.exists(os.path.join(model_dir, 'ctgan_model.pth')):
            ctgan = CTGAN(
                embedding_dim=self.config['ctgan']['embedding_dim'],
                generator_dim=self.config['ctgan']['generator_dim'],
                discriminator_dim=self.config['ctgan']['discriminator_dim'],
                device=self.device
            )
            ctgan.load_model(os.path.join(model_dir, 'ctgan_model.pth'))
            self.models['ctgan'] = ctgan
            print("CTGAN model loaded")
        
        # Load GraphSAGE
        if os.path.exists(os.path.join(model_dir, 'graphsage_model.pth')):
            graphsage = GraphSAGE(
                input_dim=self.config['graphsage']['input_dim'],
                hidden_dim=self.config['graphsage']['hidden_dim'],
                output_dim=self.config['graphsage']['output_dim'],
                num_layers=self.config['graphsage']['num_layers'],
                dropout=self.config['graphsage']['dropout']
            ).to(self.device)
            
            trainer = GraphSAGETrainer(graphsage, device=self.device)
            trainer.load_model(os.path.join(model_dir, 'graphsage_model.pth'))
            self.models['graphsage'] = graphsage
            self.models['graphsage_trainer'] = trainer
            print("GraphSAGE model loaded")
        
        # Load Cox PH
        if os.path.exists(os.path.join(model_dir, 'coxph_model.pth')):
            coxph = CoxPHModel(
                input_dim=self.config['coxph']['input_dim'],
                hidden_dims=self.config['coxph']['hidden_dims']
            )
            
            trainer = CoxPHTrainer(coxph, device=self.device)
            trainer.load_model(os.path.join(model_dir, 'coxph_model.pth'))
            self.models['coxph'] = coxph
            self.models['coxph_trainer'] = trainer
            print("Cox PH model loaded")
    
    def evaluate_ctgan(self, real_data: pd.DataFrame, 
                      synthetic_data: pd.DataFrame) -> Dict:
        """
        Evaluate CTGAN synthetic data quality
        
        Args:
            real_data: real data
            synthetic_data: synthetic data
        
        Returns:
            evaluation metrics
        """
        print("\n" + "="*50)
        print("Evaluating CTGAN Synthetic Data Quality")
        print("="*50)
        
        metrics = {}
        
        # Statistical similarity tests
        for column in real_data.columns:
            if real_data[column].dtype in [np.float64, np.int64]:
                # Kolmogorov-Smirnov test for continuous variables
                stat, p_value = ks_2samp(real_data[column], synthetic_data[column])
                metrics[f'{column}_ks_stat'] = stat
                metrics[f'{column}_ks_pvalue'] = p_value
        
        # Distribution comparison
        metrics['mean_diff'] = np.abs(real_data.mean() - synthetic_data.mean()).mean()
        metrics['std_diff'] = np.abs(real_data.std() - synthetic_data.std()).mean()
        
        print(f"Mean difference: {metrics['mean_diff']:.4f}")
        print(f"Std difference: {metrics['std_diff']:.4f}")
        
        self.evaluation_results['ctgan'] = metrics
        return metrics
    
    def evaluate_graphsage(self, test_data: pd.DataFrame, 
                          feature_names: List[str]) -> Dict:
        """
        Evaluate GraphSAGE embeddings quality
        
        Args:
            test_data: test data
            feature_names: feature names
        
        Returns:
            evaluation metrics
        """
        print("\n" + "="*50)
        print("Evaluating GraphSAGE Embeddings")
        print("="*50)
        
        if 'graphsage_trainer' not in self.models:
            print("GraphSAGE trainer not found, skipping evaluation")
            return {}
        
        trainer = self.models['graphsage_trainer']
        
        # Prepare features
        features = test_data[feature_names].values
        labels = test_data[self.config['target_column']].values
        
        # Generate embeddings
        embeddings = trainer.get_embeddings(features)
        
        # Evaluate embedding quality
        metrics = {}
        
        # Intra-class compactness
        metrics['intra_class_distance'] = self._compute_intra_class_distance(
            embeddings, labels
        )
        
        # Inter-class separation
        metrics['inter_class_distance'] = self._compute_inter_class_distance(
            embeddings, labels
        )
        
        # Separability ratio
        metrics['separability_ratio'] = (
            metrics['inter_class_distance'] / (metrics['intra_class_distance'] + 1e-10)
        )
        
        print(f"Intra-class distance: {metrics['intra_class_distance']:.4f}")
        print(f"Inter-class distance: {metrics['inter_class_distance']:.4f}")
        print(f"Separability ratio: {metrics['separability_ratio']:.4f}")
        
        self.evaluation_results['graphsage'] = metrics
        return metrics
    
    def evaluate_coxph(self, test_data: pd.DataFrame, 
                      feature_names: List[str]) -> Dict:
        """
        Evaluate Cox PH model performance
        
        Args:
            test_data: test data
            feature_names: feature names
        
        Returns:
            evaluation metrics
        """
        print("\n" + "="*50)
        print("Evaluating Cox PH Model")
        print("="*50)
        
        if 'coxph_trainer' not in self.models:
            print("Cox PH trainer not found, skipping evaluation")
            return {}
        
        trainer = self.models['coxph_trainer']
        
        # Prepare features
        features = test_data[feature_names].values
        
        # Generate survival data (if not present)
        if 'survival_time' not in test_data.columns:
            np.random.seed(self.config['random_state'])
            risk_scores = test_data[self.config['target_column']].values
            survival_time = np.random.exponential(
                scale=36 / (1 + risk_scores),
                size=len(test_data)
            )
            event = np.random.binomial(1, 0.3, size=len(test_data))
        else:
            survival_time = test_data['survival_time'].values
            event = test_data['event'].values
        
        # Predict risk scores
        risk_scores = trainer.predict_risk(features)
        
        # Compute metrics
        metrics = {}
        
        # Concordance index
        metrics['c_index'] = concordance_index(risk_scores, survival_time, event)
        
        # Risk score statistics
        metrics['risk_mean'] = risk_scores.mean()
        metrics['risk_std'] = risk_scores.std()
        metrics['risk_min'] = risk_scores.min()
        metrics['risk_max'] = risk_scores.max()
        
        # Event vs non-event risk comparison
        event_risk = risk_scores[event == 1]
        non_event_risk = risk_scores[event == 0]
        
        metrics['event_risk_mean'] = event_risk.mean() if len(event_risk) > 0 else 0
        metrics['non_event_risk_mean'] = non_event_risk.mean() if len(non_event_risk) > 0 else 0
        
        print(f"C-index: {metrics['c_index']:.4f}")
        print(f"Event risk mean: {metrics['event_risk_mean']:.4f}")
        print(f"Non-event risk mean: {metrics['non_event_risk_mean']:.4f}")
        
        # SHAP explanations
        print("\nComputing SHAP explanations...")
        shap_values = trainer.get_shap_explanations(features[:100])
        metrics['shap_values_shape'] = shap_values.shape
        
        # Feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        metrics['feature_importance'] = dict(zip(feature_names, feature_importance))
        
        print(f"SHAP values computed: {shap_values.shape}")
        
        self.evaluation_results['coxph'] = metrics
        return metrics
    
    def evaluate_classification(self, test_data: pd.DataFrame, 
                               predictions: np.ndarray) -> Dict:
        """
        Evaluate classification performance
        
        Args:
            test_data: test data
            predictions: predicted labels
        
        Returns:
            evaluation metrics
        """
        print("\n" + "="*50)
        print("Evaluating Classification Performance")
        print("="*50)
        
        true_labels = test_data[self.config['target_column']].values
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(true_labels, predictions)
        metrics['precision'] = precision_score(true_labels, predictions, average='weighted')
        metrics['recall'] = recall_score(true_labels, predictions, average='weighted')
        metrics['f1_score'] = f1_score(true_labels, predictions, average='weighted')
        
        # ROC AUC (if probabilities available)
        if hasattr(predictions, 'predict_proba'):
            probabilities = predictions.predict_proba(test_data)[:, 1]
            metrics['roc_auc'] = roc_auc_score(true_labels, probabilities)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        self.evaluation_results['classification'] = metrics
        return metrics
    
    def _compute_intra_class_distance(self, embeddings: np.ndarray, 
                                      labels: np.ndarray) -> float:
        """Compute intra-class distance"""
        distances = []
        
        for label in np.unique(labels):
            class_embeddings = embeddings[labels == label]
            
            # Compute pairwise distances
            n_samples = len(class_embeddings)
            if n_samples > 1:
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                        distances.append(dist)
        
        return np.mean(distances) if distances else 0
    
    def _compute_inter_class_distance(self, embeddings: np.ndarray, 
                                       labels: np.ndarray) -> float:
        """Compute inter-class distance"""
        class_centers = []
        
        for label in np.unique(labels):
            class_embeddings = embeddings[labels == label]
            class_center = class_embeddings.mean(axis=0)
            class_centers.append(class_center)
        
        # Compute pairwise distances between class centers
        distances = []
        n_classes = len(class_centers)
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                dist = np.linalg.norm(class_centers[i] - class_centers[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0
    
    def run_full_evaluation(self, test_data: pd.DataFrame, 
                           feature_names: List[str]) -> Dict:
        """
        Run complete evaluation pipeline
        
        Args:
            test_data: test data
            feature_names: feature names
        
        Returns:
            all evaluation results
        """
        print("\n" + "="*50)
        print("Starting Full Evaluation Pipeline")
        print("="*50)
        
        # Evaluate CTGAN
        synthetic_path = os.path.join(self.config['output_dir'], 'synthetic_data.csv')
        if os.path.exists(synthetic_path):
            synthetic_data = pd.read_csv(synthetic_path)
            self.evaluate_ctgan(test_data, synthetic_data)
        
        # Evaluate GraphSAGE
        self.evaluate_graphsage(test_data, feature_names)
        
        # Evaluate Cox PH
        self.evaluate_coxph(test_data, feature_names)
        
        # Save evaluation results
        results_path = os.path.join(self.config['log_dir'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in self.evaluation_results.items():
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, (np.int64, np.float64)):
                        json_results[key][k] = float(v)
                    else:
                        json_results[key][k] = v
            
            json.dump(json_results, f, indent=2)
        
        print(f"\nEvaluation results saved to {results_path}")
        
        return self.evaluation_results


def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'data_path': 'credit_risk_data.csv',
        'target_column': 'default',
        'random_state': 42,
        'output_dir': 'outputs',
        'model_dir': 'models',
        'log_dir': 'logs',
        'ctgan': {
            'embedding_dim': 128,
            'generator_dim': [256, 256],
            'discriminator_dim': [256, 256]
        },
        'graphsage': {
            'input_dim': 20,
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 2,
            'dropout': 0.5
        },
        'coxph': {
            'input_dim': 20,
            'hidden_dims': [128, 64]
        }
    }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate credit risk assessment models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='credit_risk_data.csv',
                       help='Path to test data file')
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
    
    # Load test data
    preprocessor = DataPreprocessor(data_path=config['data_path'],
                                   target_column=config['target_column'])
    data = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data(data)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names()
    
    # Run evaluation
    evaluator = ModelEvaluator(config)
    results = evaluator.run_full_evaluation(processed_data, feature_names)
    
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()

"""
Visualization Script for Credit Risk Assessment Model Results
Creates comprehensive visualizations for CTGAN, GraphSAGE, and Cox PH models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultVisualizer:
    """Visualizer for model results"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: configuration dictionary
        """
        self.config = config
        self.output_dir = config['output_dir']
        
        # Create visualization directory
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        print(f"Visualizer initialized. Output directory: {self.vis_dir}")
    
    def visualize_ctgan_results(self, real_data: pd.DataFrame, 
                                synthetic_data: pd.DataFrame):
        """
        Visualize CTGAN synthetic data quality
        
        Args:
            real_data: real data
            synthetic_data: synthetic data
        """
        print("\n" + "="*50)
        print("Visualizing CTGAN Results")
        print("="*50)
        
        # 1. Distribution comparison
        self._plot_distribution_comparison(real_data, synthetic_data)
        
        # 2. Correlation comparison
        self._plot_correlation_comparison(real_data, synthetic_data)
        
        # 3. PCA visualization
        self._plot_pca_comparison(real_data, synthetic_data)
        
        print("CTGAN visualizations saved!")
    
    def visualize_graphsage_results(self, embeddings: np.ndarray, 
                                    labels: np.ndarray):
        """
        Visualize GraphSAGE embeddings
        
        Args:
            embeddings: graph embeddings
            labels: node labels
        """
        print("\n" + "="*50)
        print("Visualizing GraphSAGE Results")
        print("="*50)
        
        # 1. t-SNE visualization
        self._plot_tsne_embeddings(embeddings, labels)
        
        # 2. PCA visualization
        self._plot_pca_embeddings(embeddings, labels)
        
        # 3. Embedding statistics
        self._plot_embedding_statistics(embeddings, labels)
        
        print("GraphSAGE visualizations saved!")
    
    def visualize_coxph_results(self, risk_scores: np.ndarray,
                                survival_time: np.ndarray,
                                event: np.ndarray,
                                shap_values: np.ndarray,
                                feature_names: List[str]):
        """
        Visualize Cox PH model results
        
        Args:
            risk_scores: predicted risk scores
            survival_time: survival times
            event: event indicators
            shap_values: SHAP values
            feature_names: feature names
        """
        print("\n" + "="*50)
        print("Visualizing Cox PH Results")
        print("="*50)
        
        # 1. Risk score distribution
        self._plot_risk_score_distribution(risk_scores, event)
        
        # 2. Survival curve
        self._plot_survival_curve(risk_scores, survival_time, event)
        
        # 3. SHAP summary plot
        self._plot_shap_summary(shap_values, feature_names)
        
        # 4. SHAP feature importance
        self._plot_shap_feature_importance(shap_values, feature_names)
        
        print("Cox PH visualizations saved!")
    
    def visualize_training_history(self, training_history: Dict):
        """
        Visualize training history
        
        Args:
            training_history: training history dictionary
        """
        print("\n" + "="*50)
        print("Visualizing Training History")
        print("="*50)
        
        # 1. CTGAN losses
        if 'ctgan' in training_history:
            self._plot_training_losses(training_history['ctgan']['losses'], 
                                      'CTGAN Training Losses')
        
        # 2. GraphSAGE losses
        if 'graphsage' in training_history:
            self._plot_training_losses(training_history['graphsage']['losses'],
                                      'GraphSAGE Training Losses')
        
        # 3. Cox PH losses
        if 'coxph' in training_history:
            self._plot_training_losses(training_history['coxph']['losses'],
                                      'Cox PH Training Losses')
        
        print("Training history visualizations saved!")
    
    def visualize_evaluation_metrics(self, evaluation_results: Dict):
        """
        Visualize evaluation metrics
        
        Args:
            evaluation_results: evaluation results dictionary
        """
        print("\n" + "="*50)
        print("Visualizing Evaluation Metrics")
        print("="*50)
        
        # 1. Metrics comparison
        self._plot_metrics_comparison(evaluation_results)
        
        # 2. Confusion matrix
        if 'classification' in evaluation_results:
            self._plot_confusion_matrix(evaluation_results['classification']['confusion_matrix'])
        
        print("Evaluation metrics visualizations saved!")
    
    def _plot_distribution_comparison(self, real_data: pd.DataFrame, 
                                      synthetic_data: pd.DataFrame):
        """Plot distribution comparison for each feature"""
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns
        
        n_cols = 4
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        for idx, column in enumerate(numeric_columns):
            ax = axes[idx]
            
            # Plot real data
            ax.hist(real_data[column], bins=30, alpha=0.5, 
                   label='Real', color='blue', density=True)
            
            # Plot synthetic data
            ax.hist(synthetic_data[column], bins=30, alpha=0.5,
                   label='Synthetic', color='red', density=True)
            
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution: {column}')
            ax.legend()
        
        # Remove empty subplots
        for idx in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'ctgan_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_comparison(self, real_data: pd.DataFrame, 
                                      synthetic_data: pd.DataFrame):
        """Plot correlation matrix comparison"""
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Real data correlation
        real_corr = real_data[numeric_columns].corr()
        sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Real Data Correlation Matrix')
        
        # Synthetic data correlation
        synthetic_corr = synthetic_data[numeric_columns].corr()
        sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Synthetic Data Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'ctgan_correlation_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_comparison(self, real_data: pd.DataFrame, 
                             synthetic_data: pd.DataFrame):
        """Plot PCA visualization of real vs synthetic data"""
        numeric_columns = real_data.select_dtypes(include=[np.number]).columns
        
        # Combine data
        combined_data = pd.concat([
            real_data[numeric_columns],
            synthetic_data[numeric_columns]
        ], ignore_index=True)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_data)
        
        # Create labels
        labels = ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label in ['Real', 'Synthetic']:
            mask = np.array(labels) == label
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                      label=label, alpha=0.5, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA: Real vs Synthetic Data')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'ctgan_pca_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tsne_embeddings(self, embeddings: np.ndarray, labels: np.ndarray):
        """Plot t-SNE visualization of embeddings"""
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_result = tsne.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                      label=f'Class {label}', alpha=0.5, s=50)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE Visualization of GraphSAGE Embeddings')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'graphsage_tsne_embeddings.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_embeddings(self, embeddings: np.ndarray, labels: np.ndarray):
        """Plot PCA visualization of embeddings"""
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      label=f'Class {label}', alpha=0.5, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA Visualization of GraphSAGE Embeddings')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'graphsage_pca_embeddings.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_embedding_statistics(self, embeddings: np.ndarray, labels: np.ndarray):
        """Plot embedding statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean values
        mean_values = embeddings.mean(axis=0)
        axes[0, 0].bar(range(len(mean_values)), mean_values)
        axes[0, 0].set_xlabel('Embedding Dimension')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].set_title('Mean Values per Dimension')
        
        # Standard deviation
        std_values = embeddings.std(axis=0)
        axes[0, 1].bar(range(len(std_values)), std_values)
        axes[0, 1].set_xlabel('Embedding Dimension')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Standard Deviation per Dimension')
        
        # Distribution of first two dimensions
        axes[1, 0].scatter(embeddings[:, 0], embeddings[:, 1], 
                          c=labels, alpha=0.5, cmap='viridis')
        axes[1, 0].set_xlabel('Dimension 1')
        axes[1, 0].set_ylabel('Dimension 2')
        axes[1, 0].set_title('Distribution of First Two Dimensions')
        
        # Histogram of norms
        norms = np.linalg.norm(embeddings, axis=1)
        axes[1, 1].hist(norms, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Norm')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Embedding Norms')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'graphsage_embedding_statistics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_score_distribution(self, risk_scores: np.ndarray, event: np.ndarray):
        """Plot risk score distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(risk_scores[event == 0], bins=30, alpha=0.5, 
                    label='Non-event', color='blue')
        axes[0].hist(risk_scores[event == 1], bins=30, alpha=0.5,
                    label='Event', color='red')
        axes[0].set_xlabel('Risk Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Risk Score Distribution')
        axes[0].legend()
        
        # Box plot
        data_to_plot = [risk_scores[event == 0], risk_scores[event == 1]]
        axes[1].boxplot(data_to_plot, labels=['Non-event', 'Event'])
        axes[1].set_ylabel('Risk Score')
        axes[1].set_title('Risk Score by Event Status')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'coxph_risk_distribution.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_survival_curve(self, risk_scores: np.ndarray,
                             survival_time: np.ndarray,
                             event: np.ndarray):
        """Plot survival curve by risk groups"""
        # Divide into risk groups
        n_groups = 3
        risk_groups = pd.qcut(risk_scores, n_groups, labels=['Low', 'Medium', 'High'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for group in ['Low', 'Medium', 'High']:
            mask = risk_groups == group
            group_times = survival_time[mask]
            group_events = event[mask]
            
            # Simple Kaplan-Meier estimate
            sorted_times = np.sort(group_times)
            survival_prob = np.cumprod(1 - group_events[np.argsort(group_times)] / 
                                      np.arange(1, len(group_times) + 1))
            
            ax.plot(sorted_times, survival_prob, label=f'{group} Risk', 
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Curves by Risk Groups')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'coxph_survival_curve.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_shap_summary(self, shap_values: np.ndarray, feature_names: List[str]):
        """Plot SHAP summary plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create summary plot
        for i in range(len(feature_names)):
            ax.scatter(shap_values[:, i], np.full(len(shap_values), i),
                      alpha=0.5, s=10)
        
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP Value')
        ax.set_title('SHAP Summary Plot')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'coxph_shap_summary.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_shap_feature_importance(self, shap_values: np.ndarray, 
                                       feature_names: List[str]):
        """Plot SHAP feature importance"""
        # Compute mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(sorted_features)), sorted_importance)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('SHAP Feature Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'coxph_shap_importance.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_losses(self, losses: List[float], title: str):
        """Plot training losses"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(losses, linewidth=2, alpha=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(os.path.join(self.vis_dir, filename),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, evaluation_results: Dict):
        """Plot metrics comparison across models"""
        # Extract metrics
        metrics = {}
        
        if 'ctgan' in evaluation_results:
            metrics['CTGAN'] = {
                'Mean Diff': evaluation_results['ctgan'].get('mean_diff', 0),
                'Std Diff': evaluation_results['ctgan'].get('std_diff', 0)
            }
        
        if 'graphsage' in evaluation_results:
            metrics['GraphSAGE'] = {
                'Separability': evaluation_results['graphsage'].get('separability_ratio', 0)
            }
        
        if 'coxph' in evaluation_results:
            metrics['CoxPH'] = {
                'C-index': evaluation_results['coxph'].get('c_index', 0)
            }
        
        if 'classification' in evaluation_results:
            metrics['Classification'] = {
                'Accuracy': evaluation_results['classification'].get('accuracy', 0),
                'F1-score': evaluation_results['classification'].get('f1_score', 0)
            }
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(metrics)
        
        for i, (model, model_metrics) in enumerate(metrics.items()):
            offset = (i - len(metrics)/2 + 0.5) * width
            for j, (metric_name, metric_value) in enumerate(model_metrics.items()):
                ax.bar(x[j] + offset, metric_value, width, 
                      label=model, alpha=0.7)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m for metrics_dict in metrics.values() 
                           for m in metrics_dict.keys()])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'metrics_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm: List[List[int]]):
        """Plot confusion matrix"""
        cm_array = np.array(cm)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'confusion_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'output_dir': 'outputs'
    }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = get_default_config()
    config['output_dir'] = args.output
    
    # Initialize visualizer
    visualizer = ResultVisualizer(config)
    
    # Load results
    output_dir = config['output_dir']
    
    # Load training history
    history_path = os.path.join(output_dir, '../logs/training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            training_history = json.load(f)
        visualizer.visualize_training_history(training_history)
    
    # Load evaluation results
    eval_path = os.path.join(output_dir, '../logs/evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            evaluation_results = json.load(f)
        visualizer.visualize_evaluation_metrics(evaluation_results)
    
    print("\nVisualization completed successfully!")


if __name__ == '__main__':
    main()

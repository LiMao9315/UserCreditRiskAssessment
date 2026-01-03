"""
Main Script for Credit Risk Assessment Model
Integrates CTGAN, GraphSAGE, and Cox PH models for comprehensive analysis
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import custom modules
from data_preprocessing import DataPreprocessor, CreditRiskDataset
from ctgan_model import CTGAN
from graphsage_model import GraphSAGE, CreditGraphBuilder, GraphSAGETrainer
from coxph_model import CoxPHModel, CoxPHTrainer
from train_models import ModelTrainingPipeline
from evaluate_models import ModelEvaluator
from visualize_results import ResultVisualizer


class CreditRiskPipeline:
    """Complete pipeline for credit risk assessment"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: configuration dictionary
        """
        self.config = config
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.preprocessor = None
        self.ctgan = None
        self.graphsage = None
        self.coxph = None
        
        # Results storage
        self.results = {}
        
        print("="*60)
        print("Credit Risk Assessment Pipeline Initialized")
        print("="*60)
    
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = self.config['output_dir']
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        for directory in [self.output_dir, self.model_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("Starting Full Pipeline Execution")
        print("="*60)
        
        # Step 1: Data Preprocessing
        print("\n[Step 1/5] Data Preprocessing")
        self.step_data_preprocessing()
        
        # Step 2: Model Training
        print("\n[Step 2/5] Model Training")
        self.step_model_training()
        
        # Step 3: Model Evaluation
        print("\n[Step 3/5] Model Evaluation")
        self.step_model_evaluation()
        
        # Step 4: Visualization
        print("\n[Step 4/5] Result Visualization")
        self.step_visualization()
        
        # Step 5: Generate Report
        print("\n[Step 5/5] Generate Report")
        self.step_generate_report()
        
        print("\n" + "="*60)
        print("Pipeline Execution Completed Successfully!")
        print("="*60)
    
    def step_data_preprocessing(self):
        """Step 1: Data preprocessing"""
        print("-" * 50)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(self.config['data'])
        
        # Load and preprocess data
        data = self.preprocessor.load_data()
        processed_data = self.preprocessor.preprocess_data(data)
        
        # Split data
        train_data, val_data, test_data = self.preprocessor.split_data(
            processed_data,
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size']
        )
        
        # Save preprocessed data
        self.preprocessor.save_data(train_data, 'train_data.csv')
        self.preprocessor.save_data(val_data, 'val_data.csv')
        self.preprocessor.save_data(test_data, 'test_data.csv')
        
        # Store results
        self.results['data'] = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'n_features': train_data.shape[1] - 1,  # Exclude target
            'n_classes': len(train_data['target'].unique())
        }
        
        print(f"✓ Data preprocessing completed")
        print(f"  - Train samples: {self.results['data']['train_size']}")
        print(f"  - Validation samples: {self.results['data']['val_size']}")
        print(f"  - Test samples: {self.results['data']['test_size']}")
        print(f"  - Features: {self.results['data']['n_features']}")
    
    def step_model_training(self):
        """Step 2: Model training"""
        print("-" * 50)
        
        # Initialize training pipeline
        training_pipeline = ModelTrainingPipeline(self.config)
        
        # Train all models
        training_history = training_pipeline.train_all_models()
        
        # Save training history
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Store results
        self.results['training'] = training_history
        
        print(f"✓ Model training completed")
        print(f"  - CTGAN epochs: {len(training_history['ctgan']['losses'])}")
        print(f"  - GraphSAGE epochs: {len(training_history['graphsage']['losses'])}")
        print(f"  - CoxPH epochs: {len(training_history['coxph']['losses'])}")
    
    def step_model_evaluation(self):
        """Step 3: Model evaluation"""
        print("-" * 50)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.config)
        
        # Evaluate all models
        evaluation_results = evaluator.evaluate_all_models()
        
        # Save evaluation results
        eval_path = os.path.join(self.log_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Store results
        self.results['evaluation'] = evaluation_results
        
        print(f"✓ Model evaluation completed")
        
        # Print key metrics
        if 'classification' in evaluation_results:
            cls_results = evaluation_results['classification']
            print(f"  - Classification Accuracy: {cls_results['accuracy']:.4f}")
            print(f"  - Classification F1-score: {cls_results['f1_score']:.4f}")
        
        if 'coxph' in evaluation_results:
            cox_results = evaluation_results['coxph']
            print(f"  - CoxPH C-index: {cox_results['c_index']:.4f}")
    
    def step_visualization(self):
        """Step 4: Visualization"""
        print("-" * 50)
        
        # Initialize visualizer
        visualizer = ResultVisualizer(self.config)
        
        # Visualize training history
        if 'training' in self.results:
            visualizer.visualize_training_history(self.results['training'])
        
        # Visualize evaluation metrics
        if 'evaluation' in self.results:
            visualizer.visualize_evaluation_metrics(self.results['evaluation'])
        
        print(f"✓ Visualization completed")
        print(f"  - Visualizations saved to: {visualizer.vis_dir}")
    
    def step_generate_report(self):
        """Step 5: Generate report"""
        print("-" * 50)
        
        # Generate report
        report_path = os.path.join(self.output_dir, 'report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Credit Risk Assessment Model Report\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data statistics
            if 'data' in self.results:
                f.write("1. Data Statistics\n")
                f.write("-" * 40 + "\n")
                data_stats = self.results['data']
                f.write(f"   Train samples: {data_stats['train_size']}\n")
                f.write(f"   Validation samples: {data_stats['val_size']}\n")
                f.write(f"   Test samples: {data_stats['test_size']}\n")
                f.write(f"   Number of features: {data_stats['n_features']}\n")
                f.write(f"   Number of classes: {data_stats['n_classes']}\n\n")
            
            # Training results
            if 'training' in self.results:
                f.write("2. Training Results\n")
                f.write("-" * 40 + "\n")
                training = self.results['training']
                
                f.write("   CTGAN:\n")
                f.write(f"     Final loss: {training['ctgan']['losses'][-1]:.4f}\n")
                f.write(f"     Epochs: {len(training['ctgan']['losses'])}\n\n")
                
                f.write("   GraphSAGE:\n")
                f.write(f"     Final loss: {training['graphsage']['losses'][-1]:.4f}\n")
                f.write(f"     Epochs: {len(training['graphsage']['losses'])}\n\n")
                
                f.write("   CoxPH:\n")
                f.write(f"     Final loss: {training['coxph']['losses'][-1]:.4f}\n")
                f.write(f"     Epochs: {len(training['coxph']['losses'])}\n\n")
            
            # Evaluation results
            if 'evaluation' in self.results:
                f.write("3. Evaluation Results\n")
                f.write("-" * 40 + "\n")
                evaluation = self.results['evaluation']
                
                if 'classification' in evaluation:
                    cls_results = evaluation['classification']
                    f.write("   Classification:\n")
                    f.write(f"     Accuracy: {cls_results['accuracy']:.4f}\n")
                    f.write(f"     Precision: {cls_results['precision']:.4f}\n")
                    f.write(f"     Recall: {cls_results['recall']:.4f}\n")
                    f.write(f"     F1-score: {cls_results['f1_score']:.4f}\n")
                    f.write(f"     AUC-ROC: {cls_results['auc_roc']:.4f}\n\n")
                
                if 'coxph' in evaluation:
                    cox_results = evaluation['coxph']
                    f.write("   Cox Proportional Hazards:\n")
                    f.write(f"     C-index: {cox_results['c_index']:.4f}\n\n")
                
                if 'ctgan' in evaluation:
                    ctgan_results = evaluation['ctgan']
                    f.write("   CTGAN Data Quality:\n")
                    f.write(f"     Mean difference: {ctgan_results['mean_diff']:.4f}\n")
                    f.write(f"     Std difference: {ctgan_results['std_diff']:.4f}\n\n")
                
                if 'graphsage' in evaluation:
                    graphsage_results = evaluation['graphsage']
                    f.write("   GraphSAGE:\n")
                    f.write(f"     Separability ratio: {graphsage_results['separability_ratio']:.4f}\n\n")
            
            f.write("="*60 + "\n")
            f.write("End of Report\n")
            f.write("="*60 + "\n")
        
        print(f"✓ Report generated")
        print(f"  - Report saved to: {report_path}")
    
    def save_results(self):
        """Save all results to JSON"""
        results_path = os.path.join(self.output_dir, 'results.json')
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        converted_results = convert_types(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"✓ Results saved to: {results_path}")


def get_default_config() -> dict:
    """Get default configuration"""
    return {
        'output_dir': 'outputs',
        'data': {
            'data_path': 'data/credit_risk.csv',
            'test_size': 0.2,
            'val_size': 0.1,
            'target_column': 'target',
            'categorical_columns': [],
            'numerical_columns': []
        },
        'ctgan': {
            'epochs': 100,
            'batch_size': 500,
            'latent_dim': 128,
            'generator_lr': 0.0002,
            'discriminator_lr': 0.0002
        },
        'graphsage': {
            'epochs': 100,
            'batch_size': 32,
            'hidden_dim': 64,
            'num_layers': 2,
            'learning_rate': 0.01,
            'graph_type': 'knn',
            'k_neighbors': 5
        },
        'coxph': {
            'epochs': 100,
            'batch_size': 32,
            'hidden_dim': 64,
            'learning_rate': 0.001,
            'use_neural_network': True
        },
        'training': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    }


def load_config_from_json(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Credit Risk Assessment Model - Complete Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data file'
    )
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', 'preprocess', 'train', 'evaluate', 'visualize', 'report'],
        help='Pipeline step to execute'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_json(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.output:
        config['output_dir'] = args.output
    if args.data:
        config['data']['data_path'] = args.data
    
    # Initialize pipeline
    pipeline = CreditRiskPipeline(config)
    
    # Execute pipeline
    if args.step == 'all':
        pipeline.run_full_pipeline()
        pipeline.save_results()
    elif args.step == 'preprocess':
        pipeline.step_data_preprocessing()
    elif args.step == 'train':
        pipeline.step_model_training()
    elif args.step == 'evaluate':
        pipeline.step_model_evaluation()
    elif args.step == 'visualize':
        pipeline.step_visualization()
    elif args.step == 'report':
        pipeline.step_generate_report()
    
    print("\n✓ All tasks completed successfully!")


if __name__ == '__main__':
    main()

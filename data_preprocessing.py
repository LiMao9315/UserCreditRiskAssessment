"""
Data Preprocessing Module for Credit Risk Assessment
Handles data loading, cleaning, feature engineering, and splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os


class CreditRiskDataset(Dataset):
    """PyTorch Dataset for Credit Risk Data"""
    
    def __init__(self, features, labels=None, survival_time=None, survival_event=None):
        """
        Args:
            features: numpy array or tensor of features
            labels: numpy array or tensor of labels (optional)
            survival_time: numpy array or tensor of survival times (optional)
            survival_event: numpy array or tensor of survival events (optional)
        """
        self.features = torch.FloatTensor(features) if isinstance(features, np.ndarray) else features
        self.labels = torch.FloatTensor(labels) if labels is not None and isinstance(labels, np.ndarray) else labels
        self.survival_time = torch.FloatTensor(survival_time) if survival_time is not None and isinstance(survival_time, np.ndarray) else survival_time
        self.survival_event = torch.FloatTensor(survival_event) if survival_event is not None and isinstance(survival_event, np.ndarray) else survival_event
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {'features': self.features[idx]}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        if self.survival_time is not None:
            item['survival_time'] = self.survival_time[idx]
        if self.survival_event is not None:
            item['survival_event'] = self.survival_event[idx]
        return item


class DataPreprocessor:
    """Data Preprocessing Pipeline for Credit Risk Assessment"""
    
    def __init__(self, target_col='default', survival_time_col='time', survival_event_col='event'):
        """
        Args:
            target_col: name of the target column
            survival_time_col: name of the survival time column
            survival_event_col: name of the survival event column
        """
        self.target_col = target_col
        self.survival_time_col = survival_time_col
        self.survival_event_col = survival_event_col
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        data = pd.read_csv(filepath)
        print(f"Data loaded: {data.shape}")
        return data
    
    def generate_synthetic_data(self, n_samples=10000, n_features=20, imbalance_ratio=0.1, random_state=42):
        """
        Generate synthetic credit risk data for demonstration
        
        Args:
            n_samples: total number of samples
            n_features: number of features
            imbalance_ratio: ratio of minority class (default)
            random_state: random seed
        """
        np.random.seed(random_state)
        
        # Generate features
        data = {}
        
        # Numerical features
        numerical_features = ['age', 'income', 'debt_ratio', 'credit_score', 'employment_length',
                            'loan_amount', 'interest_rate', 'monthly_payment', 'total_debt',
                            'assets_value', 'bank_balance', 'credit_utilization']
        
        for feat in numerical_features:
            if feat in ['age']:
                data[feat] = np.random.randint(18, 70, n_samples)
            elif feat in ['income', 'loan_amount', 'assets_value', 'bank_balance', 'total_debt']:
                data[feat] = np.random.exponential(scale=50000, size=n_samples)
            elif feat in ['debt_ratio', 'credit_utilization']:
                data[feat] = np.random.beta(2, 5, n_samples)
            elif feat in ['credit_score']:
                data[feat] = np.random.normal(loc=650, scale=100, size=n_samples)
                data[feat] = np.clip(data[feat], 300, 850)
            elif feat in ['employment_length']:
                data[feat] = np.random.exponential(scale=5, size=n_samples)
                data[feat] = np.clip(data[feat], 0, 40)
            elif feat in ['interest_rate']:
                data[feat] = np.random.uniform(3, 15, n_samples)
            elif feat in ['monthly_payment']:
                data[feat] = np.random.exponential(scale=1000, size=n_samples)
        
        # Categorical features
        categorical_features = ['education_level', 'employment_type', 'home_ownership', 
                              'loan_purpose', 'marital_status']
        
        education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        employment_types = ['Employed', 'Self-employed', 'Unemployed', 'Retired']
        home_ownerships = ['Rent', 'Mortgage', 'Own']
        loan_purposes = ['Personal', 'Business', 'Education', 'Medical', 'Home']
        marital_statuses = ['Single', 'Married', 'Divorced']
        
        data['education_level'] = np.random.choice(education_levels, n_samples)
        data['employment_type'] = np.random.choice(employment_types, n_samples)
        data['home_ownership'] = np.random.choice(home_ownerships, n_samples)
        data['loan_purpose'] = np.random.choice(loan_purposes, n_samples)
        data['marital_status'] = np.random.choice(marital_statuses, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate target (default) with class imbalance
        # Higher risk factors increase probability of default
        risk_score = (
            (df['debt_ratio'] * 2) +
            (df['credit_utilization'] * 1.5) +
            ((850 - df['credit_score']) / 100) +
            (df['interest_rate'] / 10) -
            (df['income'] / 100000) -
            (df['employment_length'] / 10)
        )
        
        # Normalize risk score
        risk_score = (risk_score - risk_score.mean()) / risk_score.std()
        
        # Convert to probability using sigmoid
        default_prob = 1 / (1 + np.exp(-risk_score))
        
        # Adjust for desired imbalance ratio
        threshold = np.percentile(default_prob, (1 - imbalance_ratio) * 100)
        df['default'] = (default_prob > threshold).astype(int)
        
        # Generate survival data
        # Default events have shorter survival times
        df['time'] = np.random.exponential(scale=36, size=n_samples)  # months
        df.loc[df['default'] == 1, 'time'] = np.random.exponential(scale=12, size=df['default'].sum())
        df['event'] = df['default']  # Event indicator (1 if default occurred)
        
        print(f"Synthetic data generated: {df.shape}")
        print(f"Class distribution: {df['default'].value_counts().to_dict()}")
        print(f"Imbalance ratio: {df['default'].mean():.3f}")
        
        return df
    
    def preprocess(self, data, fit=True):
        """
        Preprocess the data
        
        Args:
            data: pandas DataFrame
            fit: whether to fit the transformers (True for training data)
        
        Returns:
            processed features, labels, and survival data
        """
        df = data.copy()
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in [self.target_col, self.survival_time_col, self.survival_event_col]]
        
        if fit:
            self.feature_names = feature_cols
            self.categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
            self.numerical_features = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
        
        # Encode categorical features
        for col in self.categorical_features:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Scale numerical features
        if fit:
            df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
        else:
            df[self.numerical_features] = self.scaler.transform(df[self.numerical_features])
        
        # Extract features and targets
        features = df[feature_cols].values
        labels = df[self.target_col].values if self.target_col in df.columns else None
        survival_time = df[self.survival_time_col].values if self.survival_time_col in df.columns else None
        survival_event = df[self.survival_event_col].values if self.survival_event_col in df.columns else None
        
        return features, labels, survival_time, survival_event
    
    def split_data(self, features, labels, survival_time=None, survival_event=None, 
                   test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            features: feature array
            labels: label array
            survival_time: survival time array (optional)
            survival_event: survival event array (optional)
            test_size: proportion of test set
            val_size: proportion of validation set
            random_state: random seed
        
        Returns:
            train, val, test splits
        """
        # First split: train+val vs test
        if survival_time is not None and survival_event is not None:
            X_trainval, X_test, y_trainval, y_test, t_trainval, t_test, e_trainval, e_test = train_test_split(
                features, labels, survival_time, survival_event, 
                test_size=test_size, random_state=random_state, stratify=labels
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val, t_train, t_val, e_train, e_val = train_test_split(
                X_trainval, y_trainval, t_trainval, e_trainval,
                test_size=val_size_adjusted, random_state=random_state, stratify=y_trainval
            )
            
            return (X_train, y_train, t_train, e_train), (X_val, y_val, t_val, e_val), (X_test, y_test, t_test, e_test)
        else:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_size_adjusted, random_state=random_state, stratify=y_trainval
            )
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_dataloaders(self, train_data, val_data, test_data, batch_size=32):
        """
        Create PyTorch DataLoaders
        
        Args:
            train_data: tuple of (X_train, y_train, t_train, e_train)
            val_data: tuple of (X_val, y_val, t_val, e_val)
            test_data: tuple of (X_test, y_test, t_test, e_test)
            batch_size: batch size for DataLoader
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Check if survival data is included
        if len(train_data) == 4:
            train_dataset = CreditRiskDataset(*train_data)
            val_dataset = CreditRiskDataset(*val_data)
            test_dataset = CreditRiskDataset(*test_data)
        else:
            train_dataset = CreditRiskDataset(train_data[0], train_data[1])
            val_dataset = CreditRiskDataset(val_data[0], val_data[1])
            test_dataset = CreditRiskDataset(test_data[0], test_data[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


def save_processed_data(data_dict, output_dir='data'):
    """Save processed data to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in data_dict.items():
        if isinstance(data, np.ndarray):
            np.save(os.path.join(output_dir, f'{name}.npy'), data)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)
    
    print(f"Data saved to {output_dir}/")


def load_processed_data(input_dir='data'):
    """Load processed data from disk"""
    data_dict = {}
    
    for file in os.listdir(input_dir):
        if file.endswith('.npy'):
            name = file.replace('.npy', '')
            data_dict[name] = np.load(os.path.join(input_dir, file))
        elif file.endswith('.csv'):
            name = file.replace('.csv', '')
            data_dict[name] = pd.read_csv(os.path.join(input_dir, file))
    
    return data_dict


if __name__ == '__main__':
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Generate synthetic data
    data = preprocessor.generate_synthetic_data(n_samples=10000, imbalance_ratio=0.1)
    
    # Preprocess data
    features, labels, survival_time, survival_event = preprocessor.preprocess(data, fit=True)
    
    # Split data
    train_data, val_data, test_data = preprocessor.split_data(
        features, labels, survival_time, survival_event
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    print(f"\nData splits:")
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

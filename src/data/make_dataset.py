"""
Data loading module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handle data loading and initial preprocessing"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            self.df = self.df.sample(200000, random_state=42)
            logger.info(f"Sampled data shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def basic_cleanup(self) -> pd.DataFrame:
        """Perform basic data cleaning"""
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
            self.df = self.df.dropna()
        
        # Convert step to int
        self.df['step'] = self.df['step'].astype(int)
        
        logger.info(f"Data cleanup complete. Shape: {self.df.shape}")
        return self.df
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets"""
        X = self.df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
        y = self.df['isFraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train fraud ratio: {y_train.mean():.4f}")
        logger.info(f"Test fraud ratio: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_splits(self, X_train, X_test, y_train, y_test, output_dir: str):
        """Save train/test splits"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(output_path / 'X_train.csv', index=False)
        X_test.to_csv(output_path / 'X_test.csv', index=False)
        y_train.to_csv(output_path / 'y_train.csv', index=False)
        y_test.to_csv(output_path / 'y_test.csv', index=False)
        
        logger.info(f"Data splits saved to {output_path}")
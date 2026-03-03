"""
Feature engineering module
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create advanced features for fraud detection"""
    
    def __init__(self):
        self.feature_names = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features"""
        df = df.copy()
        
        # 1. Balance difference features
        df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['abs_balance_diff_orig'] = np.abs(df['balance_diff_orig'])
        df['abs_balance_diff_dest'] = np.abs(df['balance_diff_dest'])
        
        # 2. Amount to balance ratios (avoid division by zero)
        epsilon = 1e-5
        df['amount_to_old_balance_orig'] = df['amount'] / (df['oldbalanceOrg'] + epsilon)
        df['amount_to_new_balance_orig'] = df['amount'] / (df['newbalanceOrig'] + epsilon)
        df['amount_to_old_balance_dest'] = df['amount'] / (df['oldbalanceDest'] + epsilon)
        df['amount_to_new_balance_dest'] = df['amount'] / (df['newbalanceDest'] + epsilon)
        
        # 3. Balance error/consistency features
        df['orig_balance_error'] = np.abs(
            (df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']) / (df['amount'] + epsilon)
        )
        df['dest_balance_error'] = np.abs(
            (df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']) / (df['amount'] + epsilon)
        )
        
        # 4. Suspicious transaction patterns
        df['orig_balance_zero'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
        df['dest_balance_exact'] = (
            np.abs((df['newbalanceDest'] - df['oldbalanceDest']) - df['amount']) < epsilon
        ).astype(int)
        df['orig_start_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
        
        # 5. Transaction flags
        amount_90th = df['amount'].quantile(0.9)
        df['is_large_transaction'] = (df['amount'] > amount_90th).astype(int)
        
        # 6. Customer type features
        df['orig_customer_type'] = df['nameOrig'].astype(str).str[0].apply(
            lambda x: 1 if x == 'C' else 0
        )
        df['dest_customer_type'] = df['nameDest'].astype(str).str[0].apply(
            lambda x: 1 if x == 'C' else 0
        )
        
        # 7. Interaction features
        if 'type_encoded' in df.columns:
            df['amount_x_type'] = df['amount'] * df['type_encoded']
        df['amount_x_orig_balance'] = df['amount'] * df['oldbalanceOrg']
        
        # 8. Log transformations
        df['log_amount'] = np.log1p(df['amount'])
        df['log_oldbalanceOrg'] = np.log1p(df['oldbalanceOrg'])
        df['log_newbalanceOrig'] = np.log1p(df['newbalanceOrig'])
        df['log_oldbalanceDest'] = np.log1p(df['oldbalanceDest'])
        df['log_newbalanceDest'] = np.log1p(df['newbalanceDest'])
        
        # Define feature columns
        self.feature_names = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_diff_orig', 'balance_diff_dest', 'abs_balance_diff_orig', 'abs_balance_diff_dest',
            'amount_to_old_balance_orig', 'amount_to_new_balance_orig',
            'amount_to_old_balance_dest', 'amount_to_new_balance_dest',
            'orig_balance_error', 'dest_balance_error',
            'orig_balance_zero', 'dest_balance_exact', 'orig_start_zero',
            'is_large_transaction', 'orig_customer_type', 'dest_customer_type',
            'amount_x_orig_balance', 'log_amount', 'log_oldbalanceOrg',
            'log_newbalanceOrig', 'log_oldbalanceDest', 'log_newbalanceDest'
        ]
        
        # Add type_encoded if it exists
        if 'type_encoded' in df.columns:
            self.feature_names.append('type_encoded')
            self.feature_names.append('amount_x_type')
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Select only feature columns that exist
        existing_features = [f for f in self.feature_names if f in df.columns]
        
        logger.info(f"Created {len(existing_features)} features")
        return df[existing_features]
    
    def save_feature_names(self, output_dir: str):
        """Save feature names"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.feature_names, output_path / 'feature_names.pkl')
        logger.info(f"Feature names saved to {output_path}")
    
    def load_feature_names(self, input_dir: str):
        """Load feature names"""
        input_path = Path(input_dir) / 'feature_names.pkl'
        self.feature_names = joblib.load(input_path)
        logger.info(f"Feature names loaded from {input_path}")
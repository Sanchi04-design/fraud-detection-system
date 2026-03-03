"""
Data preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handle data preprocessing before feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_cols = None
        self.categorical_cols = None
        
    def identify_columns(self, df):
        """Identify numeric and categorical columns"""
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variables
        if 'isFraud' in self.numeric_cols:
            self.numeric_cols.remove('isFraud')
        if 'isFlaggedFraud' in self.numeric_cols:
            self.numeric_cols.remove('isFlaggedFraud')
            
        logger.info(f"Numeric columns: {self.numeric_cols}")
        logger.info(f"Categorical columns: {self.categorical_cols}")
        
    def handle_outliers(self, df, method='iqr'):
        """Handle outliers in numeric columns"""
        df_clean = df.copy()
        
        for col in self.numeric_cols:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean[col] = df[col].clip(lower_bound, upper_bound)
        
        logger.info("Outliers handled")
        return df_clean
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        df_encoded[col + '_encoded'] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        logger.info("Categorical variables encoded")
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """Scale numeric features"""
        cols_to_scale = [col for col in self.numeric_cols if col in df.columns]
        
        if not cols_to_scale:
            return df
        
        if fit:
            scaled_values = self.scaler.fit_transform(df[cols_to_scale])
        else:
            scaled_values = self.scaler.transform(df[cols_to_scale])
        
        df_scaled = df.copy()
        for i, col in enumerate(cols_to_scale):
            df_scaled[col + '_scaled'] = scaled_values[:, i]
        
        logger.info(f"Scaled {len(cols_to_scale)} features")
        return df_scaled
    
    def preprocess(self, df, fit=True):
        """Run complete preprocessing pipeline"""
        self.identify_columns(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical(df, fit)
        df = self.scale_features(df, fit)
        return df
    
    def save_preprocessors(self, output_dir: str):
        """Save preprocessors"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, output_path / 'preprocessor_scaler.pkl')
        joblib.dump(self.label_encoders, output_path / 'label_encoders.pkl')
        logger.info(f"Preprocessors saved to {output_path}")
    
    def load_preprocessors(self, input_dir: str):
        """Load preprocessors"""
        input_path = Path(input_dir)
        
        self.scaler = joblib.load(input_path / 'preprocessor_scaler.pkl')
        self.label_encoders = joblib.load(input_path / 'label_encoders.pkl')
        logger.info(f"Preprocessors loaded from {input_path}")
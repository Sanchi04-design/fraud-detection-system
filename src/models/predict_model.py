"""
Prediction module
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Handle predictions using trained model"""
    
    def __init__(self, model_path='models/best_model.pkl', 
                 threshold=0.5):
        """
        Initialize predictor with trained model
        """
        self.model = None
        self.threshold = threshold
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def set_threshold(self, threshold):
        """Set prediction threshold"""
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")
    
    def predict(self, X):
        """Make binary predictions"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Handle different model types
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def predict_with_details(self, X):
        """Get detailed predictions with risk assessment"""
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.threshold).astype(int)
        
        results = []
        for i in range(len(X)):
            prob = probabilities[i]
            results.append({
                'fraud_probability': float(prob),
                'prediction': int(predictions[i]),
                'risk_score': int(prob * 100),
                'risk_level': self.get_risk_level(prob)
            })
        
        return results
    
    def get_risk_level(self, probability):
        """Get risk level based on probability"""
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.3:
            return 'Medium'
        else:
            return 'Low'
    
    def predict_batch(self, X, batch_size=1000):
        """Make predictions in batches"""
        n_samples = len(X)
        all_predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_predictions = self.predict(batch)
            all_predictions.extend(batch_predictions)
        
        return np.array(all_predictions)
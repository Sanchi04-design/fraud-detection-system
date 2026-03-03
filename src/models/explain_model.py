"""
SHAP explanation module
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelExplainer:
    """SHAP-based model explanation"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background):
        """Create SHAP explainer"""
        # Use subset for background
        if len(X_background) > 100:
            X_background = X_background.sample(n=100, random_state=42)
        
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP explainer created")
        
    def explain(self, X):
        """Calculate SHAP values"""
        if self.explainer is None:
            self.create_explainer(X)
        
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def plot_summary(self, X, output_path):
        """Plot SHAP summary"""
        if self.shap_values is None:
            self.explain(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            X, 
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Summary plot saved to {output_path}")
    
    def plot_feature_importance(self, output_path):
        """Plot feature importance"""
        if self.shap_values is None:
            raise ValueError("Run explain() first")
        
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': shap_importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return importance_df
    
    def explain_single(self, X_single):
        """Explain single prediction"""
        if self.explainer is None:
            self.create_explainer(pd.DataFrame([X_single], columns=self.feature_names))
        
        shap_values = self.explainer.shap_values(
            pd.DataFrame([X_single], columns=self.feature_names)
        )
        
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return explanation
    
    def plot_waterfall(self, X_single, output_path):
        """Plot waterfall for single prediction"""
        if self.explainer is None:
            self.create_explainer(pd.DataFrame([X_single], columns=self.feature_names))
        
        shap_values = self.explainer.shap_values(
            pd.DataFrame([X_single], columns=self.feature_names)
        )
        
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=X_single,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
"""
Model training module
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate fraud detection models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = 0.5
        self.results = {}
        
    def get_models(self):
        """Define models with hyperparameters"""
        models = {
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    force_col_wise=True
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        return models
    
    def train_with_cv(self, X_train, y_train, model_name, model_config):
        """Train model with cross-validation"""
        logger.info(f"Training {model_name}...")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=self.random_state)),
            ('classifier', model_config['model'])
        ])
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid={'classifier__' + key: value for key, value in model_config['params'].items()},
            cv=cv,
            scoring='recall',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix']:
                logger.info(f"{metric}: {value:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def find_optimal_threshold(self, y_test, y_pred_proba):
        """Find optimal threshold for best F1-score"""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        return best_threshold
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and select best one"""
        models = self.get_models()
        
        for model_name, model_config in models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            # Train model
            model = self.train_with_cv(X_train, y_train, model_name, model_config)
            
            # Evaluate
            metrics, y_pred, y_pred_proba = self.evaluate_model(
                model.named_steps['classifier'], X_test, y_test, model_name
            )
            
            # Find optimal threshold
            threshold = self.find_optimal_threshold(y_test, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'threshold': threshold
            }
            
            # Plot confusion matrix
            self.plot_confusion_matrix(metrics['confusion_matrix'], model_name)
        
        # Select best model (by F1-score)
        self.best_model_name = max(self.results.keys(), 
                                  key=lambda x: self.results[x]['metrics']['f1_score'])
        self.best_model = self.results[self.best_model_name]['model']
        self.best_threshold = self.results[self.best_model_name]['threshold']
        
        logger.info(f"\nBest model: {self.best_model_name}")
        return self.best_model
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'reports/figures/cm_{model_name}.png')
        plt.close()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def save_results(self, filepath: str):
        """Save evaluation results"""
        results_json = {}
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'metrics': {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in result['metrics'].items()
                    if k != 'confusion_matrix'
                },
                'threshold': float(result['threshold'])
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=4)
        logger.info(f"Results saved to {filepath}")
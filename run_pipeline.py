#!/usr/bin/env python3
"""
Main pipeline script
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.make_dataset import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import ModelTrainer
from src.models.explain_model import ModelExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dirs():
    """Create directories"""
    dirs = ['data/raw', 'data/processed', 'models', 'reports/figures']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/raw/fraud.csv')
    args = parser.parse_args()
    
    create_dirs()
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    loader = DataLoader(args.data_path)
    df = loader.load_data()
    df = loader.basic_cleanup()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = loader.split_data()
    loader.save_splits(X_train, X_test, y_train, y_test, 'data/processed')
    
    # Step 3: Preprocess
    logger.info("Step 2: Preprocessing")
    preprocessor = DataPreprocessor()
    
    # Combine for preprocessing
    train_df = X_train.copy()
    train_df['isFraud'] = y_train
    test_df = X_test.copy()
    test_df['isFraud'] = y_test
    
    train_df = preprocessor.preprocess(train_df, fit=True)
    test_df = preprocessor.preprocess(test_df, fit=False)
    preprocessor.save_preprocessors('models')
    
    # Step 4: Feature engineering
    logger.info("Step 3: Feature engineering")
    engineer = FeatureEngineer()
    
    X_train_feat = engineer.create_features(train_df)
    X_test_feat = engineer.create_features(test_df)
    engineer.save_feature_names('models')
    
    # Step 5: Train models
    logger.info("Step 4: Training models")
    trainer = ModelTrainer()
    
    # Apply SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_feat, y_train)
    
    # Train
    best_model = trainer.train_all_models(X_train_res, y_train_res, 
                                         X_test_feat, y_test)
    
    # Save
    trainer.save_model('models/best_model.pkl')
    trainer.save_results('reports/metrics.json')
    
    # Step 6: Explain
    logger.info("Step 5: Model explanation")
    explainer = ModelExplainer(
        best_model.named_steps['classifier'],
        engineer.feature_names
    )
    
    X_sample = X_test_feat.sample(n=min(100, len(X_test_feat)), random_state=42)
    explainer.explain(X_sample)
    explainer.plot_summary(X_sample, 'reports/figures/shap_summary.png')
    explainer.plot_feature_importance('reports/figures/feature_importance.png')
    
    logger.info("Pipeline complete!")

if __name__ == "__main__":
    main()
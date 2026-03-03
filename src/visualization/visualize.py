"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Create visualizations"""
    
    @staticmethod
    def plot_class_distribution(y, title="Class Distribution", output_path=None):
        """Plot class distribution"""
        plt.figure(figsize=(8, 6))
        counts = pd.Series(y).value_counts()
        plt.bar(['Legitimate', 'Fraud'], counts.values, color=['#2ecc71', '#e74c3c'])
        plt.title(title)
        plt.ylabel('Count')
        
        for i, v in enumerate(counts.values):
            plt.text(i, v + 1000, str(v), ha='center')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
    
    @staticmethod
    def plot_transaction_types(df, output_path=None):
        """Plot transaction types"""
        plt.figure(figsize=(10, 6))
        df['type'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Transaction Types')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
    
    @staticmethod
    def plot_amount_distribution(df, output_path=None):
        """Plot amount distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        df['log_amount'] = np.log1p(df['amount'])
        sns.boxplot(data=df, x='isFraud', y='log_amount', ax=axes[0])
        axes[0].set_xticklabels(['Legitimate', 'Fraud'])
        axes[0].set_title('Amount Distribution (Log)')
        
        # Histogram
        for fraud in [0, 1]:
            subset = df[df['isFraud'] == fraud]
            axes[1].hist(np.log1p(subset['amount']), alpha=0.5, 
                        label='Fraud' if fraud else 'Legitimate', bins=50)
        axes[1].set_xlabel('Log Amount')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].set_title('Amount by Class')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            plt.close()
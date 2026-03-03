"""
Utility functions for Streamlit app
"""
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import random

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def get_risk_color(probability):
    """Get color based on risk probability"""
    if probability < 0.3:
        return "#28a745"  # Green
    elif probability < 0.7:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def generate_transaction_id():
    """Generate unique transaction ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_num = random.randint(1000, 9999)
    return f"TXN{timestamp}{random_num}"

def validate_input(data):
    """Validate transaction input"""
    errors = []
    
    if data.get('amount', 0) <= 0:
        errors.append("Amount must be positive")
    
    if data.get('oldbalanceOrg', 0) < 0:
        errors.append("Origin balance cannot be negative")
    
    if data.get('newbalanceOrig', 0) < 0:
        errors.append("Origin new balance cannot be negative")
    
    return len(errors) == 0, "; ".join(errors)

def get_feature_descriptions():
    """Get feature descriptions"""
    return {
        'amount': 'Transaction amount',
        'oldbalanceOrg': 'Origin account balance before',
        'newbalanceOrig': 'Origin account balance after',
        'oldbalanceDest': 'Destination balance before',
        'newbalanceDest': 'Destination balance after',
        'balance_diff_orig': 'Change in origin balance',
        'balance_diff_dest': 'Change in destination balance',
        'amount_to_balance': 'Amount to balance ratio',
        'orig_emptied': 'Origin account emptied',
        'type_encoded': 'Transaction type'
    }
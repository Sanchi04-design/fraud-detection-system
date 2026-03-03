"""
Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict_model import ModelPredictor
from src.models.explain_model import ModelExplainer
from app.utils import *

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .risk-low { background-color: #28a745; color: white; padding: 1rem; border-radius: 10px; text-align: center; }
    .risk-medium { background-color: #ffc107; color: black; padding: 1rem; border-radius: 10px; text-align: center; }
    .risk-high { background-color: #dc3545; color: white; padding: 1rem; border-radius: 10px; text-align: center; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load model artifacts"""
    artifacts = {}
    
    # Check if models exist
    if not Path("models/best_model.pkl").exists():
        st.error("Model not found. Please train first.")
        return None
    
    # Load model
    artifacts['predictor'] = ModelPredictor('models/best_model.pkl')
    
    # Load feature names
    if Path("models/feature_names.pkl").exists():
        artifacts['feature_names'] = joblib.load("models/feature_names.pkl")
    
    # Load results
    if Path("reports/metrics.json").exists():
        with open("reports/metrics.json") as f:
            artifacts['results'] = json.load(f)
    
    return artifacts

def create_input_form():
    """Create input form"""
    st.sidebar.markdown("## Transaction Details")
    
    with st.sidebar.form("transaction_form"):
        trans_type = st.selectbox(
            "Type",
            ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        )
        
        amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0)
        
        st.markdown("### Origin Account")
        nameOrig = st.text_input("Account ID", value="C123456789")
        oldbalanceOrg = st.number_input("Old Balance", min_value=0.0, value=5000.0)
        newbalanceOrig = st.number_input("New Balance", min_value=0.0, value=4000.0)
        
        st.markdown("### Destination Account")
        nameDest = st.text_input("Destination ID", value="M987654321")
        oldbalanceDest = st.number_input("Old Balance (Dest)", min_value=0.0, value=3000.0)
        newbalanceDest = st.number_input("New Balance (Dest)", min_value=0.0, value=4000.0)
        
        submitted = st.form_submit_button("Analyze")
        
    return {
        'submitted': submitted,
        'type': trans_type,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🛡️ AI Fraud Detection System</h1>
            <p>Real-time transaction risk assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load artifacts
    artifacts = load_artifacts()
    if not artifacts:
        st.stop()
    
    # Create form
    input_data = create_input_form()
    
    if input_data['submitted']:
        with st.spinner("Analyzing transaction..."):
            # Create DataFrame
            df = pd.DataFrame([{
                'type': input_data['type'],
                'amount': input_data['amount'],
                'nameOrig': input_data['nameOrig'],
                'oldbalanceOrg': input_data['oldbalanceOrg'],
                'newbalanceOrig': input_data['newbalanceOrig'],
                'nameDest': input_data['nameDest'],
                'oldbalanceDest': input_data['oldbalanceDest'],
                'newbalanceDest': input_data['newbalanceDest']
            }])
            
            # Feature engineering (simplified for demo)
            df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
            df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            df['amount_to_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)
            df['orig_emptied'] = ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
            df['type_encoded'] = pd.Categorical(df['type']).codes
            
            # Select features
            feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest',
                           'balance_diff_orig', 'balance_diff_dest',
                           'amount_to_balance', 'orig_emptied', 'type_encoded']
            
            X = df[feature_cols]
            
            # Predict
            results = artifacts['predictor'].predict_with_details(X)[0]
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Fraud Probability</h3>
                        <div style="font-size: 2rem;">{results['fraud_probability']:.1%}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Risk Score</h3>
                        <div style="font-size: 2rem;">{results['risk_score']}/100</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                risk_class = f"risk-{results['risk_level'].lower()}"
                st.markdown(f"""
                    <div class="{risk_class}">
                        <h3>Risk Level</h3>
                        <div style="font-size: 2rem;">{results['risk_level']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Amount</h3>
                        <div style="font-size: 2rem;">${input_data['amount']:,.0f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results['risk_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': '#28a745'},
                        {'range': [30, 70], 'color': '#ffc107'},
                        {'range': [70, 100], 'color': '#dc3545'}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison
            if 'results' in artifacts:
                st.subheader("Model Performance")
                results_df = pd.DataFrame(artifacts['results']).T
                metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
                plot_df = results_df[metrics].reset_index().melt(id_vars='index')
                
                fig = px.bar(plot_df, x='index', y='value', color='variable',
                           barmode='group', title="Model Comparison")
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
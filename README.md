# 🛡️ AI-Powered Fraud Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B)](https://streamlit.io/)

## 📋 Overview

Production-ready fraud detection system using the PaySim dataset. Implements industry-standard ML practices including advanced feature engineering, handling class imbalance, model comparison, hyperparameter tuning, and explainable AI with SHAP.

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/fraud-detection-system
cd fraud-detection-system
pip install -r requirements.txt

# Place fraud.csv in data/raw/

# Run pipeline
python run_pipeline.py

# Launch dashboard
streamlit run app/app.py
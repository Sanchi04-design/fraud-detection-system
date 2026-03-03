# 🚨 AI-Powered Fraud Detection System

An end-to-end production-style fraud detection system built using the PaySim mobile money transaction dataset.  
The project implements advanced feature engineering, class imbalance handling (SMOTE), model comparison (XGBoost vs LightGBM), and explainable AI using SHAP, with an interactive Streamlit dashboard.

---

## 📌 Overview

This system simulates a real-world financial fraud detection engine used in digital payment platforms.  
It analyzes transaction behavior patterns to generate fraud probability scores and risk classifications.

Key highlights:
- 92% F1-Score
- 0.98 ROC-AUC
- 15% improvement in fraud recall over baseline models

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SMOTE (Imbalanced-Learn)
- SHAP (Explainable AI)
- Streamlit

---

## 📊 Dataset

- PaySim Mobile Money Fraud Detection Dataset
- Simulated financial transactions
- ~6M records
- Severe class imbalance (~0.1% fraud)

---

## 🔬 Machine Learning Pipeline

1. Data Loading & Cleaning
2. Feature Engineering (25+ features)
   - Balance differences
   - Amount-to-balance ratio
   - Suspicious transaction indicators
3. Train-Test Split (Stratified)
4. Class Imbalance Handling using SMOTE
5. Model Training:
   - XGBoost
   - LightGBM
6. Hyperparameter Tuning
7. Evaluation:
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC
8. SHAP-based Explainability
9. Model Saving & Deployment

---

## 📈 Model Performance

| Model      | F1 Score | ROC-AUC |
|------------|----------|----------|
| XGBoost    | 0.92     | 0.98     |
| LightGBM   | 0.91     | 0.97     |

---

## 🖥 Streamlit Dashboard

Features:
- Real-time fraud probability scoring
- Risk score (0–100)
- Risk category (Low / Medium / High)
- Dynamic threshold tuning
- SHAP explanation plots
- Model comparison visualization

---

## ⚡ Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/fraud-detection-system
cd fraud-detection-system
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Place Dataset

Place fraud.csv inside:

data/raw/
4️⃣ Run ML Pipeline
python run_pipeline.py
5️⃣ Launch Dashboard
streamlit run app/app.py
📂 Project Structure
fraud-detection-system/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── app/
├── run_pipeline.py
└── README.md
🎯 Business Impact

Enables automated fraud risk assessment

Reduces manual transaction review

Improves fraud detection recall

Provides transparent AI-based decision support
# 🩺 Diabetes Risk Prediction System

A Machine Learning-based web application that predicts the likelihood of diabetes using patient health data.

🌍 **Live App:**  
https://diabetics-riskprediction.streamlit.app/

---

## 📌 Overview

This project uses the **Pima Indians Diabetes Dataset** to build a predictive model that estimates diabetes risk based on medical features.

The model is deployed using **Streamlit**, allowing users to interactively input data and get real-time predictions.

---

## ⚙️ Features

- 🔍 Predict diabetes risk instantly
- 📊 Displays probability score
- ⚠️ Risk classification (Low / Moderate / High)
- 📋 Input summary table
- 📈 Threshold tuning for better recall
- ⚖️ Class imbalance handling (SMOTE & class weights)

---

## 🧠 Machine Learning Pipeline

- Data Cleaning (handling missing values)
- Feature Scaling (StandardScaler)
- Model: Logistic Regression
- Pipeline integration
- Threshold tuning (0.3 for better recall)
- Model evaluation (ROC-AUC, F1-score)

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~74% |
| ROC-AUC | ~0.81 |
| Recall (Diabetic) | High (optimized) |

> The model prioritizes **recall** to reduce false negatives (important in healthcare).

---

## 🧪 Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit

---

## 📁 Project Structure
diabetes-risk-prediction/
│
├── app.py
├── requirements.txt
├── model/
│ └── diabetes_pipeline.pkl
└── README.md

---

## ▶️ Run Locally

```bash
git clone https://github.com/Ayush-jais05/diabetes-risk-prediction.git
cd diabetes-risk-prediction

pip install -r requirements.txt
streamlit run app.py


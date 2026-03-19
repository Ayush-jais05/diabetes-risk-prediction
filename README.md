<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/SMOTE-Imbalanced_Learn-6DB33F?style=for-the-badge" />

<br/>
<br/>

# 🩺 Diabetes Risk Prediction System

**A machine learning web app that predicts diabetes likelihood from patient health data — with real-time risk classification and probability scoring.**

[![Live App](https://img.shields.io/badge/🌐_Live_App-Click_to_Open-FF4B4B?style=for-the-badge)](https://diabetics-riskprediction.streamlit.app/)

[Overview](#-overview) · [Features](#-features) · [ML Pipeline](#-ml-pipeline) · [Performance](#-model-performance) · [Installation](#-run-locally) · [Structure](#-project-structure)

</div>

---

## 📌 Overview

This project uses the **Pima Indians Diabetes Dataset** to build a clinically-oriented predictive model that estimates diabetes risk based on key medical features.

Deployed via **Streamlit**, the app allows users to interactively enter patient data and receive instant risk assessments — designed with a **recall-first approach** to minimize dangerous false negatives.

> ⚕️ *In healthcare ML, missing a diabetic case (false negative) is far more costly than a false alarm. This model is tuned accordingly.*

---

## ✨ Features

- ⚡ **Instant prediction** from patient health inputs
- 📊 **Probability score** with confidence display
- ⚠️ **3-tier risk classification** — Low / Moderate / High
- 📋 **Input summary table** for review before prediction
- 🎛️ **Threshold tuning** (set to 0.3) to maximize recall
- ⚖️ **Class imbalance handling** via SMOTE + class weights

---

## 🧠 ML Pipeline

```
Raw Data → Cleaning → Feature Scaling → SMOTE → Model → Threshold Tuning → Prediction
```

| Step | Detail |
|---|---|
| **Data Cleaning** | Handles missing/zero values in medical features |
| **Feature Scaling** | StandardScaler for normalized input |
| **Imbalance Handling** | SMOTE oversampling + class weight adjustment |
| **Model** | Logistic Regression (interpretable, production-ready) |
| **Pipeline** | Scikit-learn Pipeline for clean train/inference parity |
| **Threshold** | Tuned to **0.3** to optimize recall for diabetic class |
| **Evaluation** | ROC-AUC, F1-Score, Confusion Matrix |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Accuracy | ~74% |
| ROC-AUC | ~0.81 |
| Recall (Diabetic) | High ✅ (optimized) |

> The model deliberately trades some precision for **higher recall** — ensuring fewer diabetic patients are missed during screening.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data & ML | Pandas, NumPy, Scikit-learn |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Deployment | Streamlit |
| Model Persistence | Joblib / Pickle |

---

## 📁 Project Structure

```
diabetes-risk-prediction/
│
├── app.py                        # Streamlit UI & prediction logic
├── requirements.txt              # Dependencies
├── README.md
│
└── model/
    └── diabetes_pipeline.pkl     # Trained sklearn pipeline
```

---

## ▶️ Run Locally

### Prerequisites
- Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/Ayush-jais05/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔮 Roadmap

- [ ] Add explainability layer (SHAP values per prediction)
- [ ] Expand dataset with additional health markers
- [ ] Multi-model comparison (Random Forest, XGBoost)
- [ ] Patient history tracking across sessions
- [ ] Docker containerization for portable deployment

---

## 👨‍💻 Author

**Ayush Raj**  
ML project focused on applying healthcare-aware modeling practices — prioritizing recall, handling class imbalance, and building interpretable, deployable pipelines.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

If this project helped you, consider giving it a ⭐ on GitHub!

</div>
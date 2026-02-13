import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
pipeline = joblib.load("model/diabetes_pipeline.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("### Predict the likelihood of diabetes using Machine Learning")
st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧾 Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

# -----------------------------
# Prediction
# -----------------------------
THRESHOLD = 0.5

if st.button("🔍 Predict"):

    # Use DataFrame (fixes sklearn warning)
    input_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [bp],
        "SkinThickness": [skin],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    prob = pipeline.predict_proba(input_df)[0][1]

    # -----------------------------
    # Risk Levels (Improved Logic)
    # -----------------------------
    if prob < 0.3:
        risk = "Low"
        st.success("✅ Low Risk - Maintain a healthy lifestyle")
    elif prob < 0.7:
        risk = "Moderate"
        st.warning("⚠️ Moderate Risk - Monitor health, improve diet & exercise")
    else:
        risk = "High"
        st.error("🚨 High Risk - Please consult a healthcare professional")

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("🧾 Prediction Result")

    st.markdown(f"### Risk Level: **{risk}**")
    st.markdown(f"### Probability of Diabetes: **{prob*100:.2f}%**")

    # Progress Bar
    st.progress(float(prob))

    # -----------------------------
    # Input Summary (NEW 🔥)
    # -----------------------------
    st.markdown("### 📋 Your Input Summary")
    st.dataframe(input_df)

# -----------------------------
# Disclaimer (VERY IMPORTANT)
# -----------------------------
st.markdown("---")
st.info("⚠️ This is a machine learning prediction tool and not a medical diagnosis. Always consult a qualified healthcare professional.")

# -----------------------------
# Footer
# -----------------------------
st.caption("⚡ Built with Machine Learning (Logistic Regression + Pipeline + Threshold Tuning)")

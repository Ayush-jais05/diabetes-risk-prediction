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
# Prediction Settings
# -----------------------------
THRESHOLD = 0.3  # tuned threshold

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict"):

    # Convert to DataFrame (fix warning)
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

    # Prediction probability
    prob = pipeline.predict_proba(input_df)[0][1]
    pred = 1 if prob > THRESHOLD else 0

    # -----------------------------
    # Risk Levels
    # -----------------------------
    if prob < 0.3:
        risk = "Low"
        color = "green"
        message = "✅ Low Risk - Maintain a healthy lifestyle"
    elif prob < 0.7:
        risk = "Moderate"
        color = "orange"
        message = "⚠️ Moderate Risk - Monitor health & improve lifestyle"
    else:
        risk = "High"
        color = "red"
        message = "🚨 High Risk - Please consult a doctor"

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("🧾 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Risk Level", risk)

    with col2:
        st.metric("Probability", f"{prob*100:.2f}%")

    # Progress Bar
    st.progress(float(prob))

    # Message
    if risk == "High":
        st.error(message)
    elif risk == "Moderate":
        st.warning(message)
    else:
        st.success(message)

    # -----------------------------
    # Input Summary
    # -----------------------------
    st.markdown("### 📋 Your Input Summary")
    st.dataframe(input_df, use_container_width=True)

    # -----------------------------
    # Model Info (NEW 🔥)
    # -----------------------------
    st.info("Model: Logistic Regression | Threshold: 0.3 | Class Imbalance Handling Applied")

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.warning("⚠️ This is NOT a medical diagnosis. Please consult a healthcare professional.")

# -----------------------------
# Footer
# -----------------------------
st.caption("🚀 Built by Ayush Raj | Machine Learning Project | Streamlit Deployment")

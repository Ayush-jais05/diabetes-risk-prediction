import streamlit as st
import numpy as np
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

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    prob = pipeline.predict_proba(input_data)[0][1]

    # Risk Levels
    if prob < 0.3:
        risk = "Low"
        color = "green"
    elif prob < 0.6:
        risk = "Medium"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("🧾 Prediction Result")

    st.markdown(f"### Risk Level: :{color}[{risk}]")
    st.markdown(f"### Probability of Diabetes: {prob*100:.2f}%")

    # Progress Bar
    st.progress(float(prob))

    # -----------------------------
    # Health Advice
    # -----------------------------
    if risk == "High":
        st.error("⚠️ High risk detected. Please consult a healthcare professional immediately.")
    elif risk == "Medium":
        st.warning("⚠️ Moderate risk. Consider improving diet, exercise, and regular monitoring.")
    else:
        st.success("✅ Low risk. Maintain a healthy lifestyle.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("⚡ Built with Machine Learning (Logistic Regression + Threshold Tuning)")

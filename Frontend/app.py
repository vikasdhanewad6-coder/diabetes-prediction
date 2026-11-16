"""
Diabetes Prediction - Streamlit frontend (STARTER VERSION).

This file loads a saved model from Model/diabetes_model.joblib and predicts
the class for values typed by the user (for example, an examiner changing
Glucose or BMI).

ASSUMPTION: the saved model is a scikit-learn Pipeline (or model) that accepts
a pandas DataFrame with exactly the FEATURES columns below, in this order.
We will confirm this in the training phase. If our final features differ,
update the FEATURES list and the input widgets.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Message that must be visible with every result.
DISCLAIMER = (
    "This is a machine learning prediction for educational purposes only. "
    "It is NOT a clinical diagnosis. Please consult a qualified doctor."
)

# Build the model path from this file location (works on laptop and cloud).
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Model" / "diabetes_model.joblib"

# Feature names (Pima Indians Diabetes dataset). Order must match training.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")


@st.cache_resource
def load_model(path: Path):
    """Load the model once and reuse it (faster app)."""
    return joblib.load(path)


st.title("Diabetes Prediction System")
st.warning(DISCLAIMER)

# If the model file is not ready yet, explain clearly and stop the app.
if not MODEL_PATH.exists():
    st.error(
        "Model file not found at Model/diabetes_model.joblib. "
        "Train and save the model first, then reload this page."
    )
    st.stop()

model = load_model(MODEL_PATH)

st.subheader("Enter values")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1, 1)
    glucose = st.number_input("Glucose (mg/dL)", 1, 300, 120, 1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70, 1)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20, 1)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80, 1)
    bmi = st.number_input("BMI", 1.0, 70.0, 25.0, 0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
    age = st.number_input("Age (years)", 1, 120, 30, 1)

if st.button("Predict"):
    # One-row table with the same column names used in training.
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=FEATURES,
    )

    predicted_class = int(model.predict(input_df)[0])

    st.subheader("Result")
    if predicted_class == 1:
        st.error("Predicted class: 1 (Diabetic outcome)")
    else:
        st.success("Predicted class: 0 (Non-diabetic outcome)")

    # Show probability only if the model supports it.
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0][1])
        st.write(f"Model probability for class 1: {prob:.2%}")

    st.write("Values used for this prediction:")
    st.dataframe(input_df)

    st.warning(DISCLAIMER)

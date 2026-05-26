import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Diabetes Predictor", page_icon="🌿", layout="centered")
st.title("🌿 Diabetes Prediction App")
st.write("Uses a **Decision Tree Classifier** trained on the Pima Indian Diabetes Dataset.")

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/decision_tree_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Model not found. Please run the Jupyter notebook first to train and save the model.")
    st.stop()

model, feature_names = load_model()

# ── Sidebar: Dataset preview ─────────────────────────────────────────────────
st.sidebar.header("📂 Dataset Preview")
data_path = "data/diabetes.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.sidebar.write(f"Shape: {df.shape}")
    st.sidebar.dataframe(df.head(5))

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("🔢 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.001)
    age = st.number_input("Age (years)", 1, 120, 30)

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🌿 Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **Result: Diabetes Detected**")
    else:
        st.success(f"✅ **Result: No Diabetes Detected**")

    col_a, col_b = st.columns(2)
    col_a.metric("No Diabetes", f"{proba[0]*100:.1f}%")
    col_b.metric("Diabetes", f"{proba[1]*100:.1f}%")

    st.caption("⚠️ This is an educational tool. Not a substitute for medical advice.")

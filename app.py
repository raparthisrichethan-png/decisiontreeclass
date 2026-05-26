import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config("KNN Classification with Diabetes Dataset", layout="wide")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass

# Title and description
st.markdown(
    """
<div class="card">
<h1>🏥 KNN Classification: Diabetes Prediction</h1>
<p>Predict whether a person has diabetes using <b>KNN Classification</b> with optimized hyperparameters</p>
</div>
""",
    unsafe_allow_html=True,
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join("data", "diabetes.csv"))

df = load_data()

# Dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Preview")
st.write(f"**Dataset shape:** {df.shape[0]} samples × {df.shape[1]} features")
st.dataframe(df.head(10))
st.markdown("</div>", unsafe_allow_html=True)

# Load saved artifacts
model_dir = os.path.join("models")
scaler_path = os.path.join(model_dir, "scaler.pkl")
model_path = os.path.join(model_dir, "knn_classifier.pkl")
feature_names_path = os.path.join(model_dir, "feature_names.pkl")
hyperparams_path = os.path.join(model_dir, "hyperparameters.pkl")

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)
feature_names = joblib.load(feature_names_path)
hyperparams_dict = joblib.load(hyperparams_path)

# Prepare data
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Display hyperparameters
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚙️ Model Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("K Neighbors", hyperparams_dict['best_params']['n_neighbors'])
with col2:
    st.metric("Weights", hyperparams_dict['best_params']['weights'])
with col3:
    st.metric("Metric", hyperparams_dict['best_params']['metric'])
st.markdown("</div>", unsafe_allow_html=True)

# Display metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Training Accuracy", f"{train_accuracy:.4f}")
with col2:
    st.metric("Testing Accuracy", f"{test_accuracy:.4f}")
with col3:
    st.metric("Precision", f"{precision:.4f}")
with col4:
    st.metric("Recall", f"{recall:.4f}")

col5, col6 = st.columns(2)
with col5:
    st.metric("F1 Score", f"{f1:.4f}")
with col6:
    st.metric("ROC-AUC", f"{roc_auc:.4f}")
st.markdown("</div>", unsafe_allow_html=True)

# Confusion Matrix
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar_kws={'label': 'Count'},
            ax=ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# ROC Curve
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# Feature Distribution
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Feature Statistics by Outcome")

selected_feature = st.selectbox("Select Feature", feature_names)
fig, ax = plt.subplots(figsize=(8, 5))
df[df['Outcome'] == 0][selected_feature].hist(bins=30, alpha=0.6, label='No Diabetes', ax=ax, color='skyblue')
df[df['Outcome'] == 1][selected_feature].hist(bins=30, alpha=0.6, label='Diabetes', ax=ax, color='coral')
ax.set_title(f'Distribution of {selected_feature}')
ax.set_xlabel(selected_feature)
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# Prediction on new data
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Diabetes Risk Prediction")
st.write("Enter patient measurements to predict diabetes risk:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)
with col2:
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=250.0, value=120.0)
with col3:
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=150.0, value=70.0)
with col4:
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)

col5, col6, col7, col8 = st.columns(4)
with col5:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=850.0, value=80.0)
with col6:
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0)
with col7:
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
with col8:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)

if st.button("🏥 Predict Diabetes Risk"):
    # Create input array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display result
    if prediction == 0:
        st.success(f"✅ **Low Risk of Diabetes** - Confidence: {prediction_proba[0]:.2%}")
    else:
        st.warning(f"⚠️ **High Risk of Diabetes** - Confidence: {prediction_proba[1]:.2%}")
    
    # Show probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Diabetes Probability", f"{prediction_proba[0]:.4f}")
    with col2:
        st.metric("Diabetes Probability", f"{prediction_proba[1]:.4f}")

st.markdown("</div>", unsafe_allow_html=True)
import streamlit as st
import numpy as np
import joblib

import joblib

log_model = joblib.load("models/logistic.pkl")
knn = joblib.load("models/knn.pkl")
svm = joblib.load("models/svm.pkl")
dt = joblib.load("models/decision_tree.pkl")
scaler = joblib.load("models/scaler.pkl")


st.set_page_config(page_title="Asteroid Hazard Prediction", layout="centered")

st.title("☄️ Asteroid Hazard Prediction System")
st.write(
    "This application predicts whether a Near-Earth Object (NEO) "
    "is potentially hazardous using machine learning models."
)

st.header("Enter Asteroid Parameters")

absolute_magnitude = st.number_input(
    "Absolute Magnitude (H)", min_value=10.0, max_value=35.0, value=22.0
)

diameter_min = st.number_input(
    "Estimated Diameter Min (m)", min_value=1.0, value=100.0
)

diameter_max = st.number_input(
    "Estimated Diameter Max (m)", min_value=1.0, value=200.0
)

relative_velocity = st.number_input(
    "Relative Velocity (km/s)", min_value=0.1, value=20.0
)

miss_distance = st.number_input(
    "Miss Distance (AU)", min_value=0.0001, value=0.05
)

model_choice = st.selectbox(
    "Choose ML Model",
    ("Logistic Regression", "KNN", "SVM", "Decision Tree")
)

input_data = np.array([[
    absolute_magnitude,
    diameter_min,
    diameter_max,
    relative_velocity,
    miss_distance
]])

# Scale if required
input_scaled = scaler.transform(input_data)

if model_choice == "Logistic Regression":
    model = log_model
    prob = model.predict_proba(input_scaled)[0][1]

elif model_choice == "KNN":
    model = knn
    prob = model.predict_proba(input_scaled)[0][1]

elif model_choice == "SVM":
    model = svm
    prob = model.predict_proba(input_scaled)[0][1]

else:
    model = dt
    prob = model.predict_proba(input_data)[0][1]

prediction = "Hazardous" if prob >= 0.5 else "Not Hazardous"

st.subheader("Prediction Result")
st.write(f"**Asteroid is:** {prediction}")
st.write(f"**Hazard Probability:** {prob:.2f}")

if prediction == "Hazardous":
    st.error("⚠️ Potentially Hazardous Asteroid!")
else:
    st.success("✅ Asteroid is Not Hazardous")

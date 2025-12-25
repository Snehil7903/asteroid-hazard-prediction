import streamlit as st
import numpy as np
import pandas as pd
import joblib

import joblib

log_model = joblib.load("models/logistic.pkl")
knn = joblib.load("models/knn.pkl")
svm = joblib.load("models/svm.pkl")
dt = joblib.load("models/decision_tree.pkl")
scaler = joblib.load("models/scaler.pkl")


st.set_page_config(page_title="Asteroid Hazard Prediction", layout="centered")

st.title("Asteroid Hazard Prediction System")
st.write(
    "This application predicts whether a Near-Earth Object (NEO) "
    "is potentially hazardous using machine learning models."
)


st.info(
    "Tap the >> icon (top-left) to open the input panel and enter asteroid parameters."
)

st.sidebar.header("☄️ Asteroid Parameters")

absolute_magnitude = st.sidebar.number_input(
    "Absolute Magnitude (H)", min_value=10.0, max_value=35.0, value=22.0
)

diameter_min = st.sidebar.number_input(
    "Estimated Diameter Min (m)", min_value=1.0, value=100.0
)

diameter_max = st.sidebar.number_input(
    "Estimated Diameter Max (m)", min_value=1.0, value=200.0
)

relative_velocity = st.sidebar.number_input(
    "Relative Velocity (km/s)", min_value=0.1, value=20.0
)

miss_distance = st.sidebar.number_input(
    "Miss Distance (AU)",
    min_value=0.00001,
    max_value=5.0,
    value=0.05,
    step=0.00001,
    format="%.5f"
)


st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    ("Logistic Regression", "KNN", "SVM", "Decision Tree")
)

st.markdown("Model Insight")

model_descriptions = {
    "Logistic Regression": "Linear probabilistic model, fast and interpretable.",
    "KNN": "Similarity-based model using nearest neighbors.",
    "SVM": "Margin-based classifier, effective for complex boundaries.",
    "Decision Tree": "Rule-based model, easy to interpret."
}




st.info(model_descriptions[model_choice])


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

st.markdown("Prediction Confidence")

st.progress(float(prob))
st.write(f"Confidence Level: **{prob*100:.2f}%**")



prediction = "Hazardous" if prob >= 0.5 else "Not Hazardous"

st.markdown("Prediction Result")

if prob >= 0.7:
    st.error("⚠️ HIGH RISK: Potentially Hazardous Asteroid")
elif prob >= 0.4:
    st.warning("⚠️ MODERATE RISK: Monitor Carefully")
else:
    st.success("✅ LOW RISK: Not Hazardous")

st.caption(
    "Note: Miss distance and absolute magnitude have the strongest influence on risk prediction."
)

st.markdown("Model Performance (ROC-AUC)")

performance_data = {
    "Model": ["Logistic Regression", "KNN", "SVM", "Decision Tree"],
    "ROC-AUC": [0.91, 0.88, 0.93, 0.87]  # replace with your actual values
}

st.table(pd.DataFrame(performance_data))

with st.expander("Machine Learning Models Used"):
    st.markdown("""
    - **Logistic Regression:** Provides smooth probability estimates and is used for risk confidence.
    - **KNN:** Classifies asteroids based on similarity to historical cases.
    - **SVM:** A conservative classifier that focuses on safe vs unsafe separation.
    - **Decision Tree:** Rule-based model offering interpretability but hard decisions.
    """)

with st.expander("Parameter Descriptions"):
    st.markdown("""
    **Absolute Magnitude (H):**  
    A measure of an asteroid’s brightness. Lower values indicate larger asteroids.

    **Estimated Diameter (Min / Max):**  
    The possible size range of the asteroid in meters, based on NASA estimates.

    **Relative Velocity (km/s):**  
    The speed of the asteroid relative to Earth. Higher velocity increases impact energy.

    **Miss Distance (AU):**  
    The closest distance between the asteroid and Earth during approach.  
    Distances ≤ 0.05 AU are considered potentially hazardous by NASA.
    """)

with st.expander("Why Predictions Change"):
    st.write(
        "Predictions change when asteroid parameters cross learned decision "
        "boundaries. Miss distance has the strongest influence, while size and "
        "velocity amplify risk. Small changes near NASA thresholds can "
        "significantly affect the predicted hazard level."
    )

with st.expander("NASA Hazard Criteria"):
    st.write(
        "According to NASA, an asteroid is considered potentially hazardous if "
        "its miss distance is less than or equal to 0.05 AU and its absolute "
        "magnitude (H) is less than 22, indicating a sufficiently large object "
        "making a close approach to Earth."
    )


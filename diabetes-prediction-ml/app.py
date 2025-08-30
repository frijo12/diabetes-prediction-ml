import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load("diabetes_model.pkl")

st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient data to predict diabetes risk.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ The model predicts this patient is **Diabetic**.")
    else:
        st.success("✅ The model predicts this patient is **Not Diabetic**.")
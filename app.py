import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
log_model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app layout
st.title("Diabetes Prediction Model")
st.write("This is a logistic regression model to predict whether a person has diabetes.")

# Collect user input through Streamlit widgets
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=0, max_value=60, value=25)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Create a DataFrame from the input data
user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree],
    'Age': [age]
})

# Scale the data using the saved scaler
user_data_scaled = scaler.transform(user_data)

# Add a Prediction Button
if st.button("Predict"):
    # Prediction using the loaded model
    prediction = log_model.predict(user_data_scaled)
    prediction_proba = log_model.predict_proba(user_data_scaled)  # Get probabilities
    
    # Get the probability of class 1 (diabetes)
    diabetes_prob = prediction_proba[0][1] * 100  # Percentage for class 1 (diabetes)
    
    # Display the result
    if prediction == 1:
        st.write(f"### Prediction: The person has diabetes.")
    else:
        st.write(f"### Prediction: The person does not have diabetes.")
    
    st.write(f"### Probability of having diabetes: {diabetes_prob:.2f}%")

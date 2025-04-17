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


# ---------- IDRS SECTION ----------
st.markdown("---")
st.subheader("Indian Diabetes Risk Score (IDRS)")

# Collect required inputs for IDRS
st.write("Please provide additional details to calculate your IDRS score:")

waist_circumference = st.number_input("Waist Circumference (cm)", min_value=50, max_value=150, value=85)
physical_activity = st.selectbox("Daily Physical Activity", ['Vigorous (exercise/manual labor)', 'Moderate (walking)', 'None'])
family_history = st.selectbox("Family History of Diabetes", ['None', 'One parent', 'Both parents'])
hypertension = st.selectbox("Do you have hypertension (high blood pressure)?", ['No', 'Yes'])
heart_history = st.selectbox("History of Heart Disease/Stroke", ['No', 'Yes'])
education = st.selectbox("Your Highest Education Level", ['No formal', 'Primary', 'Secondary', 'Graduate or above'])
income = st.selectbox("Your Monthly Income Level", ['< ₹10,000', '₹10,000–30,000', '> ₹30,000'])
caste = st.selectbox("Social Group", ['General', 'OBC', 'SC/ST', 'Other'])

# Add Predict IDRS button
if st.button("Predict IDRS"):
    # ---- Score Calculation using Weighted System ----
    idrs_score = 0

    # Age (Max = 11.42)
    if age < 35:
        idrs_score += 0
    elif age < 50:
        idrs_score += 5
    elif age < 60:
        idrs_score += 8
    else:
        idrs_score += 11.42

    # Waist Circumference (Max = 13.70)
    if waist_circumference < 80:
        idrs_score += 0
    elif waist_circumference < 90:
        idrs_score += 6
    else:
        idrs_score += 13.70

    # Physical Activity (Max = 9.13)
    activity_map = {'Vigorous (exercise/manual labor)': 0, 'Moderate (walking)': 5, 'None': 9.13}
    idrs_score += activity_map[physical_activity]

    # Family History (Max = 11.42)
    family_map = {'None': 0, 'One parent': 5, 'Both parents': 11.42}
    idrs_score += family_map[family_history]

    # BMI (Max = 11.87)
    if bmi < 23:
        idrs_score += 0
    elif bmi < 25:
        idrs_score += 5
    else:
        idrs_score += 11.87

    # Hypertension (Max = 10.04)
    idrs_score += 10.04 if hypertension == 'Yes' else 0

    # Heart Disease History (Max = 7.76)
    idrs_score += 7.76 if heart_history == 'Yes' else 0

    # Education (Max = 4.57)
    edu_map = {'No formal': 4.57, 'Primary': 3, 'Secondary': 2, 'Graduate or above': 0}
    idrs_score += edu_map[education]

    # Income (Max = 3.65)
    income_map = {'< ₹10,000': 3.65, '₹10,000–30,000': 2, '> ₹30,000': 0}
    idrs_score += income_map[income]

    # Caste (Max = 2.74)
    caste_map = {'SC/ST': 2.74, 'OBC': 1.5, 'General': 0.5, 'Other': 1}
    idrs_score += caste_map[caste]

    # Show IDRS score
    st.write(f"### Your Extended IDRS Score: `{idrs_score:.2f}` / 100")

    if idrs_score < 30:
        st.success("Low Risk of Diabetes based on IDRS.")
    elif idrs_score < 60:
        st.warning("Moderate Risk of Diabetes based on IDRS.")
    else:
        st.error("High Risk of Diabetes based on IDRS. Please consult a healthcare professional.")

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# -------------------- HEADER --------------------
col1, col2, col3 = st.columns([6, 2, 1])

with col1:
    st.markdown("<h2 style='color: #2E7D32;'>Global Next Consulting India Private Limited</h2>", unsafe_allow_html=True)

with col2:
    st.markdown(
        "<p style='text-align: center; font-size: 16px;'>"
        "<a href='https://gncipl.com/' target='_blank'>🌐 Visit our Website</a>"
        "</p>",
        unsafe_allow_html=True
    )

with col3:
    st.image("logo.jpg", width=100)

st.markdown("## 🩺 Diabetes Prediction & Indian Diabetes Risk Score (IDRS)")
st.write("This application uses machine learning to predict the likelihood of diabetes and calculates an extended Indian Diabetes Risk Score (IDRS).")

st.markdown("---")

# -------------------- LOAD MODEL --------------------
log_model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------- SESSION STATE --------------------
if "log_result" not in st.session_state:
    st.session_state["log_result"] = None
if "idrs_result" not in st.session_state:
    st.session_state["idrs_result"] = None

# -------------------- LOGISTIC REGRESSION SECTION --------------------
st.header("🔢 Logistic Regression Model Prediction")

with st.form("logistic_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    with col2:
        glucose = st.number_input("Glucose", 0, 200, 100)
        insulin = st.number_input("Insulin", 0, 900, 100)
    with col3:
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    with col4:
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 0, 120, 25)

    submitted = st.form_submit_button("Predict Diabetes")

    if submitted:
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

        user_data_scaled = scaler.transform(user_data)
        prediction = log_model.predict(user_data_scaled)
        probability = log_model.predict_proba(user_data_scaled)[0][1] * 100

        st.session_state.log_result = {
            "prediction": prediction[0],
            "probability": probability
        }

# -------------------- LOGISTIC RESULT --------------------
if st.session_state.log_result:
    st.markdown("### 🧪 Prediction Result:")
    if st.session_state.log_result["prediction"] == 1:
        st.error("🚨 The person **has diabetes**.")
    else:
        st.success("✅ The person **does not have diabetes**.")
    st.info(f"**Probability:** {st.session_state.log_result['probability']:.2f}%")

st.markdown("---")

# -------------------- IDRS SECTION --------------------
st.header("📊 Indian Diabetes Risk Score (IDRS)")

with st.form("idrs_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        waist_circumference = st.number_input("Waist Circumference (cm)", 50, 150, 85)
        physical_activity = st.selectbox("Daily Physical Activity", [
            'Vigorous (exercise/manual labor)', 'Moderate (walking)', 'None'])

    with col2:
        family_history = st.selectbox("Family History of Diabetes", ['None', 'One parent', 'Both parents'])
        hypertension = st.selectbox("Do you have hypertension?", ['No', 'Yes'])

    with col3:
        heart_history = st.selectbox("History of Heart Disease/Stroke", ['No', 'Yes'])
        education = st.selectbox("Highest Education Level", ['No formal', 'Primary', 'Secondary', 'Graduate or above'])
        income = st.selectbox("Monthly Income", ['< ₹10,000', '₹10,000–30,000', '> ₹30,000'])
        caste = st.selectbox("Social Group", ['General', 'OBC', 'SC/ST', 'Other'])

    idrs_submit = st.form_submit_button("Predict IDRS")

    if idrs_submit:
        idrs_score = 0
        idrs_score += 0 if age < 35 else 5 if age < 50 else 8 if age < 60 else 11.42
        idrs_score += 0 if waist_circumference < 80 else 6 if waist_circumference < 90 else 13.70
        idrs_score += {'Vigorous (exercise/manual labor)': 0, 'Moderate (walking)': 5, 'None': 9.13}[physical_activity]
        idrs_score += {'None': 0, 'One parent': 5, 'Both parents': 11.42}[family_history]
        idrs_score += 0 if bmi < 23 else 5 if bmi < 25 else 11.87
        idrs_score += 10.04 if hypertension == 'Yes' else 0
        idrs_score += 7.76 if heart_history == 'Yes' else 0
        idrs_score += {'No formal': 4.57, 'Primary': 3, 'Secondary': 2, 'Graduate or above': 0}[education]
        idrs_score += {'< ₹10,000': 3.65, '₹10,000–30,000': 2, '> ₹30,000': 0}[income]
        idrs_score += {'SC/ST': 2.74, 'OBC': 1.5, 'General': 0.5, 'Other': 1}[caste]

        st.session_state.idrs_result = round(idrs_score, 2)

# -------------------- IDRS RESULT --------------------
if st.session_state.idrs_result is not None:
    st.markdown("### 🧮 Your Extended IDRS Score:")
    st.markdown(f"<h2 style='color: #673AB7;'>{st.session_state.idrs_result} / 100</h2>", unsafe_allow_html=True)

    if st.session_state.idrs_result < 30:
        st.success("🟢 Low Risk of Diabetes based on IDRS.")
    elif st.session_state.idrs_result < 60:
        st.warning("🟡 Moderate Risk of Diabetes based on IDRS.")
    else:
        st.error("🔴 High Risk of Diabetes. Please consult a doctor.")


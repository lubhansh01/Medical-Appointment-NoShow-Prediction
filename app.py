import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Medical Appointment Analytics",
    layout="wide"
)

# ------------------------
# Load model & features
# ------------------------
model = joblib.load("models/noshow_model.pkl")
feature_names = joblib.load("models/noshow_features.pkl")

st.title("🏥 Medical Appointment No-Show Prediction")

st.markdown("Predict the likelihood of a patient missing an appointment.")

# ------------------------
# User Inputs (HUMAN FRIENDLY)
# ------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=35)

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender == "Male" else 0

    sms = st.selectbox("SMS Reminder Received?", ["Yes", "No"])
    sms_val = 1 if sms == "Yes" else 0

with col2:
    specialty = st.number_input("Specialty Code", min_value=0, value=1)
    place = st.number_input("City Code", min_value=0, value=1)
    shift = st.selectbox("Appointment Shift", ["Morning", "Afternoon", "Evening"])

    shift_map = {"Morning": 0, "Afternoon": 1, "Evening": 2}
    shift_val = shift_map[shift]

# ------------------------
# Build FULL feature vector (CRITICAL FIX)
# ------------------------
input_dict = {col: 0 for col in feature_names}

# Assign known values
if "age" in input_dict:
    input_dict["age"] = age
if "gender" in input_dict:
    input_dict["gender"] = gender_val
if "SMSreceived" in input_dict:
    input_dict["SMSreceived"] = sms_val
if "specialty" in input_dict:
    input_dict["specialty"] = specialty
if "place" in input_dict:
    input_dict["place"] = place
if "appointment_shift" in input_dict:
    input_dict["appointment_shift"] = shift_val

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# ------------------------
# Prediction
# ------------------------
if st.button("🔮 Predict No-Show Risk"):
    risk = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        label="No-Show Probability",
        value=f"{risk * 100:.2f}%"
    )

    if risk > 0.7:
        st.error("🔴 High Risk: Strong chance of no-show")
    elif risk > 0.4:
        st.warning("🟡 Medium Risk: Consider reminder")
    else:
        st.success("🟢 Low Risk: Patient likely to attend")

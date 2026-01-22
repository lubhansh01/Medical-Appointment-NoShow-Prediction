import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Medical Appointment Analytics", layout="wide")

noshow_model = joblib.load("models/noshow_model.pkl")
demand_model = joblib.load("models/demand_model.pkl")

st.title("🏥 Medical Appointment Analytics Dashboard")

tab1, tab2 = st.tabs(["🧍 No-Show Predictor", "📊 Demand Forecast"])

# No Show Tab

with tab1:
    st.subheader("No-Show Risk Prediction")

    age = st.number_input("Age", 0, 100, 30)
    sms = st.selectbox("SMS Received", [0,1])
    gender = st.selectbox("Gender (Encoded)", [0,1])
    specialty = st.number_input("Specialty Code", 0, 50, 1)
    place = st.number_input("City Code", 0, 50, 1)

    if st.button("Predict Risk"):
        X = [[age, gender, specialty, place, sms, 1, 1, 1]]
        risk = noshow_model.predict_proba(X)[0][1]
        st.progress(risk)
        st.success(f"No-Show Risk: {risk*100:.2f}%")

# Forecast Tab

with tab2:
    st.subheader("Demand Forecast")

    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    weekday = st.slider("Weekday", 0, 6, 2)

    if st.button("Predict Demand"):
        pred = demand_model.predict([[day, month, weekday]])
        st.success(f"Expected Appointments: {int(pred[0])}")

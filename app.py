import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Medical Appointment Analytics",
    layout="wide"
)

# -------------------------
# Load models
# -------------------------
noshow_model = joblib.load("models/noshow_model.pkl")
demand_model = joblib.load("models/demand_model.pkl")

st.title("🏥 Medical Appointment No-Show & Demand Forecasting")

tab1, tab2 = st.tabs(["🧍 No-Show Prediction", "📊 Demand Forecasting"])

# =========================================================
# TAB 1: NO-SHOW PREDICTION
# =========================================================
with tab1:
    st.subheader("Predict Appointment No-Show Risk")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=35)
        gender = st.selectbox("Gender (Encoded)", [0, 1])
        sms = st.selectbox("SMS Received", [0, 1])

    with col2:
        specialty = st.number_input("Specialty Code", min_value=0, value=1)
        place = st.number_input("City Code", min_value=0, value=1)
        shift = st.number_input("Appointment Shift Code", min_value=0, value=1)

    if st.button("Predict No-Show Risk"):
        with st.spinner("Predicting..."):
            input_data = np.array([[age, gender, specialty, place, shift, sms]])
            risk = noshow_model.predict_proba(input_data)[0][1]

        st.success(f"🧠 No-Show Risk: **{risk*100:.2f}%**")

        if risk > 0.7:
            st.error("🔴 High Risk: Consider reminder or rescheduling")
        elif risk > 0.4:
            st.warning("🟡 Medium Risk")
        else:
            st.info("🟢 Low Risk")

# =========================================================
# TAB 2: DEMAND FORECASTING
# =========================================================
with tab2:
    st.subheader("Forecast Appointment Demand")

    col1, col2, col3 = st.columns(3)

    with col1:
        day = st.slider("Day", 1, 31, 15)
    with col2:
        month = st.slider("Month", 1, 12, 6)
    with col3:
        weekday = st.slider("Weekday (0=Mon)", 0, 6, 2)

    if st.button("Predict Demand"):
        with st.spinner("Forecasting..."):
            prediction = demand_model.predict([[day, month, weekday]])

        st.success(f"📈 Expected Appointments: **{int(prediction[0])}**")

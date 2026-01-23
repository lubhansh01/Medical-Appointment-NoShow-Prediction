import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Medical Appointment Analytics",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# Load Models & Feature Schema
# =====================================================
noshow_model = joblib.load("models/noshow_model.pkl")
noshow_features = joblib.load("models/noshow_features.pkl")
demand_model = joblib.load("models/demand_model.pkl")

# =====================================================
# Global Styling (UI ONLY – NO LOGIC CHANGE)
# =====================================================
st.markdown("""
<style>

/* Base font */
html, body, [class*="css"] {
    font-size: 18px;
}

/* Main title */
.main-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 0;
}

/* Subtitle */
.subtitle {
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Section title */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Card container */
.card {
    background-color: #0f172a;
    padding: 22px;
    border-radius: 14px;
    margin-top: 15px;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #2563eb, #22c55e);
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 10px 26px;
    border-radius: 12px;
    border: none;
}

/* Remove unnecessary dividers */
hr {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# Header
# =====================================================
st.markdown('<div class="main-title">🏥 Medical Appointment Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered prediction of appointment no-shows and demand forecasting</div>',
    unsafe_allow_html=True
)

# =====================================================
# Tabs
# =====================================================
tab1, tab2 = st.tabs(["🧍 No-Show Prediction", "📊 Demand Forecasting"])

# =====================================================
# TAB 1: NO-SHOW PREDICTION
# =====================================================
with tab1:
    st.markdown('<div class="section-title">Appointment No-Show Risk Prediction</div>', unsafe_allow_html=True)

    # Dropdown mappings (UI-friendly → numeric)
    CITY_MAP = {
        "Delhi (1)": 1,
        "Mumbai (2)": 2,
        "Bengaluru (3)": 3,
        "Chennai (4)": 4
    }

    SPECIALTY_MAP = {
        "General Physician (1)": 1,
        "Cardiologist (2)": 2,
        "Orthopedic (3)": 3,
        "Pediatrician (4)": 4
    }

    SHIFT_MAP = {
        "Morning": 0,
        "Afternoon": 1,
        "Evening": 2
    }

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sms = st.selectbox("SMS Reminder Received?", ["Yes", "No"])

    with col2:
        city_label = st.selectbox("City", list(CITY_MAP.keys()))
        specialty_label = st.selectbox("Doctor Specialty", list(SPECIALTY_MAP.keys()))
        shift_label = st.selectbox("Appointment Shift", list(SHIFT_MAP.keys()))

    gender_val = 1 if gender == "Male" else 0
    sms_val = 1 if sms == "Yes" else 0
    city_val = CITY_MAP[city_label]
    specialty_val = SPECIALTY_MAP[specialty_label]
    shift_val = SHIFT_MAP[shift_label]

    if st.button("🔮 Predict No-Show Risk"):
        # Build input with EXACT feature names
        input_data = {feature: 0 for feature in noshow_features}

        if "age" in input_data:
            input_data["age"] = age
        if "gender" in input_data:
            input_data["gender"] = gender_val
        if "SMSreceived" in input_data:
            input_data["SMSreceived"] = sms_val
        if "place" in input_data:
            input_data["place"] = city_val
        if "specialty" in input_data:
            input_data["specialty"] = specialty_val
        if "appointment_shift" in input_data:
            input_data["appointment_shift"] = shift_val

        input_df = pd.DataFrame([input_data])

        risk = noshow_model.predict_proba(input_df)[0][1]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.metric("No-Show Probability", f"{risk*100:.2f}%")

        if risk > 0.7:
            st.error("🔴 High Risk: Patient very likely to miss appointment")
        elif risk > 0.4:
            st.warning("🟡 Medium Risk: Reminder recommended")
        else:
            st.success("🟢 Low Risk: Patient likely to attend")

        st.markdown("""
        **How to use this prediction**
        - High Risk → Call / Reschedule
        - Medium Risk → Send reminder
        - Low Risk → Normal scheduling
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# TAB 2: DEMAND FORECASTING
# =====================================================
with tab2:
    st.markdown('<div class="section-title">Appointment Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown("Predict expected number of appointments for a specific day.")

    col1, col2, col3 = st.columns(3)

    with col1:
        day = st.slider("Day of Month", 1, 31, 15)
    with col2:
        month = st.slider("Month", 1, 12, 6)
    with col3:
        weekday = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    if st.button("📈 Predict Demand"):
        prediction = demand_model.predict([[day, month, weekday_map[weekday]]])

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Expected Appointments", int(prediction[0]))
        st.info("Helps hospitals plan staff, slots, and resources efficiently.")
        st.markdown('</div>', unsafe_allow_html=True)

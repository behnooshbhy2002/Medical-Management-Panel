"""
Streamlit Dashboard — Intelligent Diabetes Risk Analysis System
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://api_gateway:8000"

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Diabetes Risk Analysis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Right-to-left support is removed (optional — keep if needed for multilingual)
# st.markdown("""<style> body { direction: rtl; } </style>""", unsafe_allow_html=True)

st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; font-size: 1.6em; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.6em; }
    .risk-low    { color: #27ae60; font-weight: bold; font-size: 1.6em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x80?text=Health+AI", width=200)
st.sidebar.title("🏥 Menu")

page = st.sidebar.radio(
    "Select Page:",
    [
        "🔬 Diabetes Risk Prediction",
        "📊 Population Statistics",
        "💊 Prescription Analysis",
        "📋 Prediction History"
    ]
)

# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────
def api_call(method: str, endpoint: str, data: dict = None):
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        else:
            r = requests.post(url, json=data, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "❌ Could not connect to API Gateway"
    except requests.HTTPError as e:
        return None, f"❌ Server error: {e.response.status_code} — {e.response.text[:120]}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def risk_emoji(level: str):
    emojis = {
        "Low": "🟢 Low",
        "Medium": "🟡 Medium",
        "High": "🔴 High",
        "Very High": "🔴🔴 Very High"
    }
    return emojis.get(level, "⚪ Unknown")


# ─────────────────────────────────────────
# Page 1: Diabetes Risk Prediction
# ─────────────────────────────────────────
if page == "🔬 Diabetes Risk Prediction":
    st.title("🔬 Diabetes Risk Prediction")
    st.markdown("Enter patient information to calculate the estimated diabetes risk.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧑 Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 45)
        smoking = st.selectbox(
            "Smoking History",
            ["never", "No Info", "former", "current", "ever", "not current"]
        )

    with col2:
        st.subheader("🩺 Clinical Information")
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1)
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.8, step=0.1)
        glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=50, max_value=400, value=120)
        hypertension = st.checkbox("Has Hypertension")
        heart_disease = st.checkbox("Has Heart Disease")

    if st.button("🔍 Calculate Risk", use_container_width=True, type="primary"):
        payload = {
            "gender": gender,
            "age": float(age),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "smoking_history": smoking,
            "bmi": bmi,
            "hba1c_level": hba1c,
            "blood_glucose_level": float(glucose),
        }

        with st.spinner("Calculating risk..."):
            result, err = api_call("POST", "/predict/diabetes", payload)

        if err:
            st.error(err)
        else:
            st.divider()
            c1, c2, c3 = st.columns(3)

            prob = result["diabetes_probability"]
            level = result["risk_level"]

            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Diabetes Probability (%)"},
                    number={'font': {'size': 38}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.3 else "#27ae60"},
                        "steps": [
                            {"range": [0, 30],   "color": "#d5f5e3"},
                            {"range": [30, 60],  "color": "#fef9e7"},
                            {"range": [60, 100], "color": "#fadbd8"},
                        ],
                    }
                ))
                fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.metric("Risk Level", risk_emoji(level))
                st.metric("Probability", result["diabetes_probability_percent"])
                st.info(result["recommendation"])

            with c3:
                st.subheader("⚠️ Key Risk Factors")
                factors = result.get("risk_factors", [])
                if factors:
                    for f in factors:
                        st.error(f"• {f}")
                else:
                    st.success("No major risk factors identified")


# ─────────────────────────────────────────
# Page 2: Population Statistics
# ─────────────────────────────────────────
elif page == "📊 Population Statistics":
    st.title("📊 Diabetes Population Statistics")

    with st.spinner("Loading statistics..."):
        data, err = api_call("GET", "/stats/diabetes")

    if err:
        st.error(err)
    else:
        ov = data["overview"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", f"{ov['total_patients']:,}")
        col2.metric("Diabetic Patients", f"{ov['diabetic_count']:,}")
        col3.metric("Non-Diabetic", f"{ov['healthy_count']:,}")
        col4.metric("Prevalence Rate", f"{ov['diabetes_prevalence_percent']}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📏 BMI Comparison")
            bmi_data = pd.DataFrame({
                "Group": ["Diabetic", "Non-Diabetic"],
                "Mean BMI": [data["bmi"]["diabetic_mean"], data["bmi"]["healthy_mean"]]
            })
            fig = px.bar(
                bmi_data,
                x="Group",
                y="Mean BMI",
                color="Group",
                color_discrete_sequence=["#e74c3c", "#27ae60"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🩸 HbA1c Comparison")
            hba1c_data = pd.DataFrame({
                "Group": ["Diabetic", "Non-Diabetic"],
                "Mean HbA1c": [data["hba1c"]["diabetic_mean"], data["hba1c"]["healthy_mean"]]
            })
            fig = px.bar(
                hba1c_data,
                x="Group",
                y="Mean HbA1c",
                color="Group",
                color_discrete_sequence=["#e74c3c", "#27ae60"]
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("👥 Age Distribution")
        age_groups = data["age"]["age_distribution"]
        age_df = pd.DataFrame({
            "Age Group": list(age_groups.keys()),
            "Count": list(age_groups.values()),
        })
        fig = px.pie(age_df, names="Age Group", values="Count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# Page 3: Prescription Analysis
# ─────────────────────────────────────────
elif page == "💊 Prescription Analysis":
    st.title("💊 Prescription & Clinical Text Analysis")

    text = st.text_area(
        "Enter prescription or clinical note:",
        height=220,
        placeholder="Patient prescribed Claritin 10mg daily for allergic rhinitis. Also taking metformin 500mg BID..."
    )

    if st.button("🔍 Analyze Prescription", use_container_width=True, type="primary"):
        if not text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result, err = api_call("POST", "/analyze/prescription", {"text": text})

            if err:
                st.error(err)
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🏥 Predicted Medical Specialty")
                    spec = result["specialty"]
                    st.success(f"**{spec['predicted_specialty']}**")
                    st.metric("Confidence", f"{spec['confidence']*100:.1f}%")
                    st.markdown("**Top 3 specialties:**")
                    for item in spec.get("top3", []):
                        st.write(f"• {item['specialty']}: {item['confidence']*100:.1f}%")

                with col2:
                    st.subheader("💊 Detected Medications")
                    drugs = result["drugs"]
                    st.metric("Number of Drugs Detected", drugs["drug_count"])

                    if drugs["drug_list"]:
                        for drug, label in drugs["detected_entities"].items():
                            st.code(f"{drug} → {label}")
                    else:
                        st.info("No medications were detected in the text.")


# ─────────────────────────────────────────
# Page 4: History
# ─────────────────────────────────────────
elif page == "📋 Prediction History":
    st.title("📋 Prediction & Analysis History")

    with st.spinner("Loading history..."):
        data, err = api_call("GET", "/history?limit=50")

    if err:
        st.error(err)
    elif data:
        df = pd.DataFrame(data)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            df[["id", "type", "created_at"]].rename(columns={
                "id": "ID",
                "type": "Prediction Type",
                "created_at": "Date & Time"
            }),
            use_container_width=True
        )
    else:
        st.info("No prediction history available yet.")
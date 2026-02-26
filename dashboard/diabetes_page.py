import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go

from config import API_URL, Q_DIABETES, STREAM_INTERVAL
from helpers import http_post, consume_one, risk_icon, stream_controls, append_log

def show_diabetes_page():
    st.title("🔬 Diabetes Risk Analysis")
    tab_m, tab_s = st.tabs(["✏️ Manual", "📡 Live Stream"])

    # ── Manual ───────────────────────────────────────────────
    with tab_m:
        st.subheader("Enter patient data")
        c1, c2 = st.columns(2)
        with c1:
            gender  = st.selectbox("Gender", ["Male", "Female", "Other"], key="dm_gender")
            age     = st.slider("Age", 1, 100, 45, key="dm_age")
            smoking = st.selectbox("Smoking history",
                ["never", "No Info", "former", "current", "ever", "not current"],
                key="dm_smoke")
        with c2:
            bmi     = st.number_input("BMI",           10.0, 70.0, 27.5, 0.1, key="dm_bmi")
            hba1c   = st.number_input("HbA1c level",    3.0, 15.0,  5.8, 0.1, key="dm_hba1c")
            glucose = st.number_input("Blood glucose",  50,  400,  120,       key="dm_glucose")
            hyper   = st.checkbox("Hypertension",  key="dm_hyper")
            heart   = st.checkbox("Heart disease", key="dm_heart")

        if st.button("🔍 Calculate Risk", use_container_width=True, type="primary", key="dm_btn"):
            payload = {
                "gender": gender, "age": float(age),
                "hypertension": int(hyper), "heart_disease": int(heart),
                "smoking_history": smoking, "bmi": bmi,
                "hba1c_level": hba1c, "blood_glucose_level": float(glucose),
            }
            with st.spinner("Calculating..."):
                result, err = http_post(f"{API_URL}/predict/diabetes", payload)

            if err:
                st.error(err)
            else:
                st.divider()
                prob  = result["diabetes_probability"]
                level = result["risk_level"]
                r1, r2, r3 = st.columns(3)

                with r1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "Diabetes Probability (%)"},
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.3 else "#27ae60"},
                            "steps": [
                                {"range": [0,  30],  "color": "#d5f5e3"},
                                {"range": [30, 60],  "color": "#fef9e7"},
                                {"range": [60, 100], "color": "#fadbd8"},
                            ],
                        },
                    ))
                    fig.update_layout(height=280, margin=dict(t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with r2:
                    st.metric("Risk Level",  f"{risk_icon(level)} {level}")
                    st.metric("Probability", result["diabetes_probability_percent"])
                    st.info(result["recommendation"])

                with r3:
                    st.subheader("⚠️ Risk Factors")
                    for f in result.get("risk_factors", []):
                        st.error(f"• {f}")
                    if not result.get("risk_factors"):
                        st.success("No major risk factors found")

    # ── Live Stream ──────────────────────────────────────────
    with tab_s:
        st.subheader("📡 Live patient monitoring — data from queue.diabetes")
        st.caption("Producer reads diabetes_prediction_dataset.csv and publishes one row every 5 s.")

        start, stop = stream_controls("diabetes")
        if start:
            st.session_state.run_diabetes = True
            st.rerun()
        if stop:
            st.session_state.run_diabetes = False
            st.rerun()

        ph_kpi   = st.empty()
        ph_table = st.empty()
        ph_chart = st.empty()

        def render_diabetes_stream():
            log = st.session_state.log_diabetes
            if not log:
                ph_kpi.info("Press ▶️ Start to begin consuming from queue.diabetes")
                return
            df = pd.DataFrame(log)

            high_risk = df["level"].isin(["High", "Very High"]).sum()
            avg_prob  = df["prob"].mean() * 100

            with ph_kpi.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Processed patients",  len(df))
                k2.metric("High risk",            high_risk)
                k3.metric("Avg probability",      f"{avg_prob:.1f}%")
                k4.metric("Last patient",
                          f"{risk_icon(df.iloc[-1]['level'])} {df.iloc[-1]['level']}")

            show = df.tail(10)[
                ["age","bmi","hba1c_level","blood_glucose_level","level","prob"]
            ].copy()
            show.columns = ["Age","BMI","HbA1c","Glucose","Risk Level","Probability"]
            show["Probability"] = show["Probability"].apply(lambda x: f"{x*100:.1f}%")
            ph_table.dataframe(show[::-1], use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df["prob"] * 100, mode="lines+markers",
                line=dict(color="#e74c3c", width=2), marker=dict(size=5),
                name="Diabetes probability %",
            ))
            fig.add_hline(y=60, line_dash="dash", line_color="#e74c3c", annotation_text="High risk (60%)")
            fig.add_hline(y=30, line_dash="dash", line_color="#f39c12", annotation_text="Medium risk (30%)")
            fig.update_layout(
                title="Diabetes probability trend",
                yaxis_title="Probability (%)", xaxis_title="Patient #",
                height=300, margin=dict(t=40, b=30),
            )
            ph_chart.plotly_chart(fig, use_container_width=True)

        render_diabetes_stream()

        if st.session_state.run_diabetes:
            msg = consume_one(Q_DIABETES)
            if msg:
                result, err = http_post(f"{API_URL}/predict/diabetes", msg)
                if result:
                    append_log("log_diabetes", {
                        **{k: msg[k] for k in ["age","bmi","hba1c_level","blood_glucose_level"]},
                        "prob":  result["diabetes_probability"],
                        "level": result["risk_level"],
                    })
            else:
                st.toast("Queue is empty — waiting for producer...", icon="⏳")
            time.sleep(STREAM_INTERVAL)
            st.rerun()
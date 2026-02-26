# =============================================================
# FILE: dashboard/app.py
# ROLE: Streamlit dashboard.
#       - Manual tab : user enters data, calls service directly
#       - Stream tab  : consumes one message from RabbitMQ queue,
#                       sends to service, updates charts live
#
# Pages:
#   🔬 Diabetes      -> manual form  |  live stream from queue.diabetes
#   🧬 Disease       -> manual form  |  live stream from queue.disease
#   💊 Drug          -> manual form  |  live stream from queue.drug
#   📊 Stats         -> snapshot     |  live stream (polls /stats)
#   📋 History
# =============================================================

import streamlit as st
import requests
import pika
import json
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ── Service URLs ──────────────────────────────────────────────
API_URL      = os.getenv("API_URL",      "http://api_gateway:8000")
STATS_URL    = os.getenv("STATS_URL",    "http://diabetes_stats:8000")
DISEASE_URL  = os.getenv("DISEASE_URL",  "http://specialty_classifier:8000")
DRUG_URL     = os.getenv("DRUG_URL",     "http://drug_detector:8000")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")

# ── Queue names (must match producer.py) ─────────────────────
Q_DIABETES = "queue.diabetes"
Q_DISEASE  = "queue.disease"
Q_DRUG     = "queue.drug"

STREAM_INTERVAL = 5   # seconds between each rerun cycle

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Health AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { min-width: 240px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; padding: 8px 18px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────
_DEFAULTS = {
    "run_diabetes": False,
    "run_disease":  False,
    "run_drug":     False,
    "run_stats":    False,
    # stream logs
    "log_diabetes": [],   # list of {age, bmi, hba1c_level, blood_glucose_level, level, prob}
    "log_disease":  [],   # list of {text_short, ground_truth, predicted, confidence}
    "log_drug":     [],   # list of {text_short, drugs: list[str]}
    "log_stats":    [],   # list of {total, rate, bmi_d, bmi_h, hba1c_d, hba1c_h, glucose_d, glucose_h}
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def http_post(url: str, payload: dict, timeout: int = 15):
    """POST to a microservice. Returns (data, error_str)."""
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Service unavailable"
    except Exception as e:
        return None, str(e)


def http_get(url: str, timeout: int = 10):
    """GET from a microservice. Returns (data, error_str)."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Service unavailable"
    except Exception as e:
        return None, str(e)


def consume_one(queue: str) -> dict | None:
    """
    Pull exactly ONE message from a RabbitMQ queue (basic_get).
    Returns the parsed JSON payload or None if the queue is empty / unreachable.
    """
    try:
        conn    = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = conn.channel()
        channel.queue_declare(queue=queue, durable=True)

        method, _props, body = channel.basic_get(queue=queue, auto_ack=True)
        conn.close()

        if method is None:
            return None   # queue is empty
        return json.loads(body)
    except Exception as e:
        st.warning(f"RabbitMQ error: {e}")
        return None


def risk_icon(level: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Very High": "🔴🔴"}.get(level, "⚪")


def stream_controls(key: str):
    """Render ▶️ / ⏹️ buttons. Returns (start_pressed, stop_pressed)."""
    c1, c2, _ = st.columns([1, 1, 6])
    start = c1.button("▶️ Start", key=f"{key}_start", type="primary",
                      disabled=st.session_state[f"run_{key}"])
    stop  = c2.button("⏹️ Stop",  key=f"{key}_stop",
                      disabled=not st.session_state[f"run_{key}"])
    return start, stop


def append_log(key: str, entry: dict, max_len: int = 100):
    st.session_state[key].append(entry)
    if len(st.session_state[key]) > max_len:
        st.session_state[key].pop(0)

# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🏥 Health AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Section",
    ["🔬 Diabetes", "🧬 Disease", "💊 Drug", "📊 Stats", "📋 History"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Each page has two tabs:\n"
    "- **Manual**: enter data, get instant result\n"
    "- **Live Stream**: pulls from RabbitMQ every 5 s"
)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Diabetes
# ══════════════════════════════════════════════════════════════
if page == "🔬 Diabetes":
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
                            "bar":  {"color": "#e74c3c" if prob > 0.6
                                              else "#f39c12" if prob > 0.3
                                              else "#27ae60"},
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
            fig.add_hline(y=60, line_dash="dash", line_color="#e74c3c",
                          annotation_text="High risk (60%)")
            fig.add_hline(y=30, line_dash="dash", line_color="#f39c12",
                          annotation_text="Medium risk (30%)")
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
                        **{k: msg[k] for k in
                           ["age","bmi","hba1c_level","blood_glucose_level"]},
                        "prob":  result["diabetes_probability"],
                        "level": result["risk_level"],
                    })
            else:
                st.toast("Queue is empty — waiting for producer...", icon="⏳")
            time.sleep(STREAM_INTERVAL)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Disease Detection
# ══════════════════════════════════════════════════════════════
elif page == "🧬 Disease":
    st.title("🧬 Disease Detection from Patient Conversation")
    tab_m, tab_s = st.tabs(["✏️ Manual", "📡 Live Stream"])

    # ── Manual ───────────────────────────────────────────────
    with tab_m:
        st.subheader("Enter patient complaint")
        symptom_text = st.text_area(
            "Patient description:",
            height=150,
            placeholder="I have been feeling very thirsty, urinating frequently, and losing weight...",
            key="dis_text",
        )
        if st.button("🔍 Detect Disease", use_container_width=True, type="primary", key="dis_btn"):
            if not symptom_text.strip():
                st.warning("Please enter patient complaint text")
            else:
                with st.spinner("Analyzing..."):
                    result, err = http_post(f"{DISEASE_URL}/classify", {"text": symptom_text})
                if err:
                    st.error(err)
                else:
                    st.divider()
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🎯 Detection Result")
                        conf = result["confidence"] * 100
                        st.markdown(
                            f"<div style='font-size:22px;font-weight:bold;"
                            f"color:#2c3e50;padding:15px;background:#eaf4fb;"
                            f"border-radius:10px;text-align:center;'>"
                            f"🦠 {result['predicted_specialty']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric("Model confidence", f"{conf:.1f}%")
                        st.progress(result["confidence"])
                    with c2:
                        st.subheader("📊 Top 3 Candidates")
                        top3 = result.get("top3", [])
                        if top3:
                            names = [t["specialty"]          for t in top3]
                            confs = [t["confidence"] * 100 for t in top3]
                            fig = go.Figure(go.Bar(
                                x=confs, y=names, orientation="h",
                                marker_color=["#2ecc71","#f39c12","#e74c3c"],
                                text=[f"{c:.1f}%" for c in confs],
                                textposition="outside",
                            ))
                            fig.update_layout(
                                height=200, xaxis_title="Confidence (%)",
                                margin=dict(t=10, b=10, l=10, r=40),
                                xaxis=dict(range=[0, max(confs) * 1.3]),
                            )
                            st.plotly_chart(fig, use_container_width=True)

    # ── Live Stream ──────────────────────────────────────────
    with tab_s:
        st.subheader("📡 Live disease classification — data from queue.disease")
        st.caption("Producer reads medical_conversations.csv and publishes one row every 5 s.")

        start, stop = stream_controls("disease")
        if start:
            st.session_state.run_disease = True
            st.rerun()
        if stop:
            st.session_state.run_disease = False
            st.rerun()

        ph_dis_kpi   = st.empty()
        ph_dis_chart = st.empty()
        ph_dis_log   = st.empty()

        def render_disease_stream():
            log = st.session_state.log_disease
            if not log:
                ph_dis_kpi.info("Press ▶️ Start to begin consuming from queue.disease")
                return
            df = pd.DataFrame(log)

            with ph_dis_kpi.container():
                k1, k2, k3 = st.columns(3)
                k1.metric("Total processed",           len(df))
                k2.metric("Unique diseases detected",  df["predicted"].nunique())
                k3.metric("Latest prediction",
                          f"🦠 {df.iloc[-1]['predicted']} "
                          f"({df.iloc[-1]['confidence']*100:.0f}%)")

            dist = df["predicted"].value_counts().reset_index()
            dist.columns = ["Disease", "Count"]
            fig = px.bar(dist, x="Count", y="Disease", orientation="h",
                         color="Count", color_continuous_scale="Blues",
                         title="Distribution of detected diseases")
            fig.update_layout(
                height=max(250, len(dist) * 35),
                margin=dict(t=40, b=10), showlegend=False,
            )
            ph_dis_chart.plotly_chart(fig, use_container_width=True)

            show = df.tail(8)[["text_short","ground_truth","predicted","confidence"]].copy()
            show.columns = ["Patient text", "Ground truth", "Predicted", "Confidence"]
            show["Confidence"] = show["Confidence"].apply(lambda x: f"{x*100:.1f}%")
            ph_dis_log.dataframe(show[::-1], use_container_width=True, hide_index=True)

        render_disease_stream()

        if st.session_state.run_disease:
            msg = consume_one(Q_DISEASE)
            if msg:
                result, err = http_post(f"{DISEASE_URL}/classify", {"text": msg["text"]})
                if result:
                    append_log("log_disease", {
                        "text_short":  msg["text"][:60] + "...",
                        "ground_truth":msg.get("ground_truth", ""),
                        "predicted":   result["predicted_specialty"],
                        "confidence":  result["confidence"],
                    })
            else:
                st.toast("Queue is empty — waiting for producer...", icon="⏳")
            time.sleep(STREAM_INTERVAL)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Drug Detection
# ══════════════════════════════════════════════════════════════
elif page == "💊 Drug":
    st.title("💊 Drug Detection from Prescription Text")
    tab_m, tab_s = st.tabs(["✏️ Manual", "📡 Live Stream"])

    # ── Manual ───────────────────────────────────────────────
    with tab_m:
        st.subheader("Enter prescription text")
        rx_text = st.text_area(
            "Prescription:",
            height=150,
            placeholder="Patient prescribed Claritin 10mg daily. Also taking Zyrtec and Nasonex...",
            key="rx_text",
        )
        if st.button("🔍 Extract Drugs", use_container_width=True, type="primary", key="rx_btn"):
            if not rx_text.strip():
                st.warning("Please enter prescription text")
            else:
                with st.spinner("Analyzing..."):
                    result, err = http_post(f"{DRUG_URL}/detect", {"text": rx_text})
                if err:
                    st.error(err)
                else:
                    st.divider()
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("Drugs detected", result["drug_count"])
                        st.caption(f"Method: `{result.get('method_used','regex')}`")
                        st.markdown("---")
                        for drug, label in result.get("detected_entities", {}).items():
                            ca, cb = st.columns([2, 1])
                            ca.markdown(f"**💊 {drug}**")
                            cb.markdown(
                                f"<span style='background:#3498db;color:#fff;"
                                f"border-radius:4px;padding:2px 6px;font-size:11px'>"
                                f"{label}</span>",
                                unsafe_allow_html=True,
                            )
                    with c2:
                        if result["drug_list"]:
                            fig = px.bar(
                                x=result["drug_list"],
                                y=[1] * len(result["drug_list"]),
                                labels={"x": "Drug", "y": ""},
                                title="Detected drugs",
                                color=result["drug_list"],
                            )
                            fig.update_layout(
                                showlegend=False, height=280,
                                yaxis=dict(showticklabels=False),
                                margin=dict(t=40, b=10),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No drugs found in text")

    # ── Live Stream ──────────────────────────────────────────
    with tab_s:
        st.subheader("📡 Live drug extraction — data from queue.drug")
        st.caption("Producer reads mtsamples.csv and publishes one row every 5 s.")

        start, stop = stream_controls("drug")
        if start:
            st.session_state.run_drug = True
            st.rerun()
        if stop:
            st.session_state.run_drug = False
            st.rerun()

        ph_rx_kpi   = st.empty()
        ph_rx_chart = st.empty()
        ph_rx_log   = st.empty()

        def render_drug_stream():
            log = st.session_state.log_drug
            if not log:
                ph_rx_kpi.info("Press ▶️ Start to begin consuming from queue.drug")
                return

            all_drugs   = [d for row in log for d in row["drugs"]]
            drug_counts = Counter(all_drugs)

            with ph_rx_kpi.container():
                k1, k2, k3 = st.columns(3)
                k1.metric("Prescriptions processed", len(log))
                k2.metric("Total drugs detected",     len(all_drugs))
                k3.metric("Most frequent drug",
                          drug_counts.most_common(1)[0][0] if drug_counts else "—")

            if drug_counts:
                dc_df = pd.DataFrame(
                    drug_counts.most_common(10), columns=["Drug", "Count"]
                )
                fig = px.bar(dc_df, x="Drug", y="Count",
                             color="Count", color_continuous_scale="Viridis",
                             title="Top 10 most frequent drugs")
                fig.update_layout(
                    height=300, margin=dict(t=40, b=10), showlegend=False,
                )
                ph_rx_chart.plotly_chart(fig, use_container_width=True)

            show_rows = [
                {
                    "Prescription":    row["text_short"],
                    "Drugs detected":  ", ".join(row["drugs"]) if row["drugs"] else "—",
                    "Count":           len(row["drugs"]),
                }
                for row in log[-8:]
            ]
            ph_rx_log.dataframe(
                pd.DataFrame(show_rows)[::-1],
                use_container_width=True, hide_index=True,
            )

        render_drug_stream()

        if st.session_state.run_drug:
            msg = consume_one(Q_DRUG)
            if msg:
                result, err = http_post(f"{DRUG_URL}/detect", {"text": msg["text"]})
                if result:
                    append_log("log_drug", {
                        "text_short": msg["text"][:60] + "...",
                        "drugs":      result.get("drug_list", []),
                    })
            else:
                st.toast("Queue is empty — waiting for producer...", icon="⏳")
            time.sleep(STREAM_INTERVAL)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 4 — Stats
# ══════════════════════════════════════════════════════════════
elif page == "📊 Stats":
    st.title("📊 Diabetes Population Statistics")
    tab_m, tab_s = st.tabs(["✏️ Snapshot", "📡 Live Stream"])

    # ── Snapshot ─────────────────────────────────────────────
    with tab_m:
        if st.button("🔄 Load Stats", use_container_width=True, key="stat_load"):
            with st.spinner("Fetching..."):
                data, err = http_get(f"{STATS_URL}/stats")
            if err:
                st.error(err)
            else:
                ov = data["overview"]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total patients",   f"{ov['total_patients']:,}")
                k2.metric("Diabetic",          f"{ov['diabetic_count']:,}")
                k3.metric("Healthy",           f"{ov['healthy_count']:,}")
                k4.metric("Diabetes rate",     f"{ov['diabetes_prevalence_percent']}%")

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    bmi_df = pd.DataFrame({
                        "Group": ["Diabetic", "Healthy"],
                        "BMI":   [data["bmi"]["diabetic_mean"],
                                  data["bmi"]["healthy_mean"]],
                    })
                    fig = px.bar(bmi_df, x="Group", y="BMI", color="Group",
                                 color_discrete_sequence=["#e74c3c","#27ae60"],
                                 title="Average BMI comparison")
                    fig.update_layout(height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    hba_df = pd.DataFrame({
                        "Group":  ["Diabetic", "Healthy"],
                        "HbA1c":  [data["hba1c"]["diabetic_mean"],
                                   data["hba1c"]["healthy_mean"]],
                    })
                    fig = px.bar(hba_df, x="Group", y="HbA1c", color="Group",
                                 color_discrete_sequence=["#e74c3c","#27ae60"],
                                 title="Average HbA1c comparison")
                    fig.update_layout(height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                age_g  = data["age"]["age_distribution"]
                age_df = pd.DataFrame({
                    "Age group": list(age_g.keys()),
                    "Count":     list(age_g.values()),
                })
                fig = px.pie(age_df, names="Age group", values="Count",
                             hole=0.4, title="Age distribution")
                fig.update_layout(height=350, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ── Live Stream ──────────────────────────────────────────
    with tab_s:
        st.subheader("📡 Auto-refreshing stats — polls /stats every 5 s")
        st.caption(
            "The diabetes_stats service continuously receives new patients "
            "from queue.diabetes and recalculates statistics on each /stats call."
        )

        start, stop = stream_controls("stats")
        if start:
            st.session_state.run_stats = True
            st.rerun()
        if stop:
            st.session_state.run_stats = False
            st.rerun()

        ph_st_kpi    = st.empty()
        ph_st_trend  = st.empty()
        ph_st_charts = st.empty()

        def render_stats_stream(snap: dict | None = None):
            history = st.session_state.log_stats
            if not history and snap is None:
                ph_st_kpi.info("Press ▶️ Start to begin live stats")
                return
            if snap:
                history.append({
                    "total":     snap["overview"]["total_patients"],
                    "rate":      snap["overview"]["diabetes_prevalence_percent"],
                    "bmi_d":     snap["bmi"]["diabetic_mean"],
                    "bmi_h":     snap["bmi"]["healthy_mean"],
                    "hba1c_d":   snap["hba1c"]["diabetic_mean"],
                    "hba1c_h":   snap["hba1c"]["healthy_mean"],
                    "glucose_d": snap["blood_glucose"]["diabetic_mean"],
                    "glucose_h": snap["blood_glucose"]["healthy_mean"],
                })
                if len(history) > 60:
                    history.pop(0)
                st.session_state.log_stats = history

            last = history[-1]
            prev = history[-2] if len(history) > 1 else None

            with ph_st_kpi.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total patients",  f"{last['total']:,}",
                          f"+{last['total'] - prev['total']}" if prev else None)
                k2.metric("Diabetes rate",   f"{last['rate']}%",
                          f"{last['rate'] - prev['rate']:+.2f}%" if prev else None)
                k3.metric("Diabetic BMI avg",   last["bmi_d"])
                k4.metric("Diabetic HbA1c avg", last["hba1c_d"])

            hdf = pd.DataFrame(history)

            with ph_st_trend.container():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=hdf["rate"], mode="lines+markers",
                    line=dict(color="#e74c3c", width=2),
                    name="Diabetes rate %",
                ))
                fig.update_layout(
                    title="Diabetes rate over time",
                    yaxis_title="Rate (%)", height=260,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with ph_st_charts.container():
                cc1, cc2, cc3 = st.columns(3)
                for col, title, key_d, key_h in [
                    (cc1, "BMI",        "bmi_d",     "bmi_h"),
                    (cc2, "HbA1c",      "hba1c_d",   "hba1c_h"),
                    (cc3, "Blood glucose","glucose_d", "glucose_h"),
                ]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=hdf[key_d], mode="lines", name="Diabetic",
                        line=dict(color="#e74c3c", width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        y=hdf[key_h], mode="lines", name="Healthy",
                        line=dict(color="#27ae60", width=2),
                    ))
                    fig.update_layout(
                        title=f"Avg {title}",
                        height=220, margin=dict(t=35, b=15),
                        legend=dict(orientation="h", y=-0.3),
                    )
                    col.plotly_chart(fig, use_container_width=True)

        render_stats_stream()

        if st.session_state.run_stats:
            snap, err = http_get(f"{STATS_URL}/stats")
            if snap:
                render_stats_stream(snap)
            time.sleep(STREAM_INTERVAL)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 5 — History
# ══════════════════════════════════════════════════════════════
elif page == "📋 History":
    st.title("📋 Prediction History")

    c_btn, _ = st.columns([1, 5])
    if c_btn.button("🔄 Load", key="hist_load"):
        data, err = http_get(f"{API_URL}/history?limit=50")
        if err:
            st.error(err)
        elif data:
            df = pd.DataFrame(data)
            df["created_at"] = (
                pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            )
            st.dataframe(
                df[["id", "type", "created_at"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No predictions recorded yet")
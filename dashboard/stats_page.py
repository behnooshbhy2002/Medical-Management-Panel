import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pika
import json
from collections import deque
import numpy as np
from collections import defaultdict

from config import STREAM_INTERVAL, STATS_URL, RABBITMQ_URL, Q_DIABETES
from helpers import stream_controls, http_get, consume_one

def fetch_new_diabetes_messages(max_messages=15):
    new_rows = []
    try:
        params = pika.URLParameters(RABBITMQ_URL)
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.queue_declare(queue="queue.diabetes", durable=True)

        for _ in range(max_messages):
            method_frame, _, body = ch.basic_get(queue="queue.diabetes", auto_ack=True)
            if method_frame is None:
                break
            try:
                msg = json.loads(body)
                new_rows.append(msg)
            except:
                pass

        conn.close()
    except Exception as e:
        st.warning(f"RabbitMQ connection issue: {e}")
    return new_rows


def update_local_stats(rows):
    if not rows:
        return

    s = st.session_state.live_stats
    patients = st.session_state.live_patients

    for row in rows:
        patients.append(row)
        s["count_total"] += 1

        bmi     = row.get("bmi", 25.0)
        hba1c   = row.get("hba1c_level", 5.5)
        glucose = row.get("blood_glucose_level", 100.0)
        age     = row.get("age", 40.0)
        gender  = str(row.get("gender", "Male")).strip().lower()
        htn     = row.get("hypertension", 0)
        gender  = str(row.get("gender", "Male")).strip().lower()
        smoking = str(row.get("smoking_history", "No Info")).strip().lower()

        # Diabetes detection logic → you can change this or use a real model here
        is_diabetic = hba1c >= 6.5 or glucose >= 126 or (bmi >= 30 and hba1c >= 5.7)

        if is_diabetic:
            s["count_diabetic"] += 1
            s["count_d"] += 1
            s["sum_bmi_d"] += bmi
            s["sum_hba1c_d"] += hba1c
            s["sum_age_d"] += age
            if gender == "female":
                s["count_f_diab"] += 1
            if gender == "male":
                s["count_m_diab"] += 1
            if htn:
                s["count_htn_diab"] += 1
            
            # smoking
            s["smoking_diab"][smoking] += 1
            
            # age
            s["ages_diab"].append(age)
            
            # glucose
            s["sum_glucose_d"] += glucose
            s["count_glucose_d"] += 1

        else:
            s["count_h"] += 1
            s["sum_bmi_h"] += bmi

            if gender == "female":
                s["count_f_healthy"] += 1
            elif gender == "male":
                s["count_m_healthy"] += 1
            
            s["smoking_healthy"][smoking] += 1
            s["ages_healthy"].append(age)
            
            if htn:
                s["count_htn_healthy"] += 1
            
            s["sum_glucose_h"] += glucose
            s["count_glucose_h"] += 1


def compute_display_stats():
    s = st.session_state.live_stats
    n_d = s["count_d"] or 1
    n_h = s["count_h"] or 1
    n_total = s["count_total"] or 1

    return {
        "total": s["count_total"],
        "diabetic_count": s["count_diabetic"],
        "diabetic_rate": 100.0 * s["count_diabetic"] / n_total,
        "bmi_d_avg": s["sum_bmi_d"] / n_d,
        "bmi_h_avg": s["sum_bmi_h"] / n_h,
        "hba1c_d_avg": s["sum_hba1c_d"] / n_d,
        "age_d_avg": s["sum_age_d"] / n_d,
        "diab_f": s["count_f_diab"],
        "diab_m": s["count_m_diab"],
        "healthy_f": s["count_f_healthy"],
        "healthy_m": s["count_m_healthy"],
        
        "smoking_diab": dict(s["smoking_diab"]),   # for chart
        "smoking_healthy": dict(s["smoking_healthy"]),
        
        "htn_diab_pct": 100.0 * s["count_htn_diab"] / (s["count_d"] or 1),
        "htn_healthy_pct": 100.0 * s["count_htn_healthy"] / (s["count_h"] or 1),
        
        "glucose_d_avg": s["sum_glucose_d"] / (s["count_glucose_d"] or 1),
        "glucose_h_avg": s["sum_glucose_h"] / (s["count_glucose_h"] or 1),
    }

def show_stats_page():
    st.title("📊 Diabetes Population Statistics")
    if "live_patients" not in st.session_state:
        st.session_state.live_patients = deque(maxlen=5000)
        st.session_state.live_stats = {
            "count_total": 0,
            "count_diabetic": 0,
            "sum_bmi_d": 0.0,
            "sum_bmi_h": 0.0,
            "count_d": 0,
            "count_h": 0,
            "sum_hba1c_d": 0.0,
            "sum_age_d": 0.0,
            "count_f_diab": 0,
            "count_m_diab": 0,
            "count_f_healthy": 0,
            "count_m_healthy": 0,
            
            "smoking_diab": defaultdict(int),
            "smoking_healthy": defaultdict(int),
            
            "ages_diab": [],
            "ages_healthy": [],
            
            "count_htn_diab": 0,
            "count_htn_healthy": 0,
            
            "sum_glucose_d": 0.0,
            "sum_glucose_h": 0.0,
            "count_glucose_d": 0,
            "count_glucose_h": 0,
        }
        st.session_state.history = []        
        st.session_state.run_stream = False

    tab_m, tab_s = st.tabs(["✏️ Snapshot", "📡 Live Stream"])

    # ── Snapshot ─────────────────────────────────────────────
    with tab_m:
        if st.button("🔄 Load Stats", use_container_width=True, key="stat_load"):
            with st.spinner("Fetching..."):
                data, err = http_get(f"{STATS_URL}/stats")
            
            if err:
                st.error(err)
                st.stop()

            ov = data["overview"]
            
            # First row - main metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total patients",     f"{ov['total_patients']:,}")
            k2.metric("Diabetic",           f"{ov['diabetic_count']:,}")
            k3.metric("Healthy",            f"{ov['healthy_count']:,}")
            k4.metric("Diabetes rate",      f"{ov['diabetes_prevalence_percent']}%")

            st.divider()

            # Second row - important averages
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Age - Diabetic",  f"{data['age']['diabetic_mean_age']:.1f} years")
            m2.metric("Avg Age - Healthy",   f"{data['age']['healthy_mean_age']:.1f} years")
            m3.metric("Avg BMI - Diabetic",  f"{data['bmi']['diabetic_mean']:.1f}")
            m4.metric("Avg BMI - Healthy",   f"{data['bmi']['healthy_mean']:.1f}")

            st.divider()

            # Charts & distributions
            c1, c2 = st.columns(2)
            with c1:
                # BMI comparison
                bmi_df = pd.DataFrame({
                    "Group": ["Diabetic", "Healthy"],
                    "Mean BMI": [data["bmi"]["diabetic_mean"], data["bmi"]["healthy_mean"]],
                })
                fig_bmi = px.bar(bmi_df, x="Group", y="Mean BMI", color="Group",
                                 color_discrete_sequence=["#e74c3c", "#27ae60"],
                                 title="Average BMI")
                st.plotly_chart(fig_bmi, use_container_width=True)

            with c2:
                # HbA1c comparison
                hba_df = pd.DataFrame({
                    "Group": ["Diabetic", "Healthy"],
                    "Mean HbA1c": [data["hba1c"]["diabetic_mean"], data["hba1c"]["healthy_mean"]],
                })
                fig_hba = px.bar(hba_df, x="Group", y="Mean HbA1c", color="Group",
                                 color_discrete_sequence=["#e74c3c", "#27ae60"],
                                 title="Average HbA1c")
                st.plotly_chart(fig_hba, use_container_width=True)

            st.subheader("Age Distribution")
            age_dist = data["age"]["age_distribution"]
            age_df = pd.DataFrame({
                "Age Group": list(age_dist.keys()),
                "Count": list(age_dist.values()),
            })
            fig_age = px.pie(age_df, names="Age Group", values="Count",
                             hole=0.4, title="Overall Age Distribution")
            st.plotly_chart(fig_age, use_container_width=True)

            # Gender
            st.subheader("Gender Distribution - Diabetic Patients")
            gender_d = data["gender_distribution"]["diabetic_by_gender"]
            gender_df = pd.DataFrame({
                "Gender": list(gender_d.keys()),
                "Count": list(gender_d.values()),
            })
            fig_gender = px.pie(gender_df, names="Gender", values="Count",
                                title="Gender of Diabetic Patients", hole=0.3)
            fig_gender.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_gender, use_container_width=True)

            # Other percentages (comorbidities, obesity, smoking, ...)
            cols = st.columns(3)
            with cols[0]:
                st.metric("Obesity in Diabetics", f"{data['bmi']['diabetic_obesity_percent']}%")
                st.metric("Severe Obesity", f"{data['bmi']['diabetic_severe_obesity_percent']}%")
            with cols[1]:
                st.metric("Hypertension in Diabetics", f"{data['comorbidities']['diabetic_hypertension_percent']}%")
                st.metric("Heart Disease in Diabetics", f"{data['comorbidities']['diabetic_heart_disease_percent']}%")
            with cols[2]:
                st.metric("HbA1c Above Threshold", f"{data['hba1c']['diabetic_above_threshold_percent']}%")
                st.metric("Current Smokers (Diabetics)", f"{data['smoking']['diabetic_smoking_current_percent']}%")

    # ── Live Stream ──────────────────────────────────────────
    with tab_s:
        st.subheader("Live incoming patients stream")

        start, stop = stream_controls("stream")
        if start:
            st.session_state.run_stream = True
            st.rerun()
        if stop:
            st.session_state.run_stream = False
            st.rerun()

        @st.fragment(run_every=4 if st.session_state.get("run_stream", False) else None)
        def _live_panel():
            if not st.session_state.get("run_stream", False):
                st.info("▶️ Press **Start** to begin live stream.")
                return

            new_rows = fetch_new_diabetes_messages(max_messages=10)
            # new_rows = consume_one(Q_DIABETES)
            update_local_stats(new_rows)

            stats = compute_display_stats()
            s = st.session_state.live_stats

            st.session_state.history.append({
                "rate": stats["diabetic_rate"],
                "bmi_d_avg": stats["bmi_d_avg"],
                "ts": time.time(),
                "bmi_d": stats["bmi_d_avg"],
                "bmi_h": stats["bmi_h_avg"],
                "hba1c_d": stats["hba1c_d_avg"],
                "age_d": stats["age_d_avg"],
                "diab_f": stats["diab_f"],
                "diab_m": stats["diab_m"],
                "glucose_d_avg": stats["glucose_d_avg"],
            })
            if len(st.session_state.history) > 300:
                st.session_state.history.pop(0)

            df_hist = pd.DataFrame(st.session_state.history)

            # KPI
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total", f"{stats['total']:,}")
            k2.metric("Diabetic %", f"{stats['diabetic_rate']:.1f}%")
            k3.metric("Avg BMI (D)", f"{stats['bmi_d_avg']:.1f}")
            k4.metric("Avg HbA1c (D)", f"{stats['hba1c_d_avg']:.1f}")

            # Diabetes rate trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_hist["rate"], mode="lines", line_color="#e74c3c"))
            fig.update_layout(height=280, title="Diabetes Rate Trend")
            st.plotly_chart(fig, use_container_width=True, key="chart_trend")

            # ── Lower charts ────────
            c_gen, c_smoke, c_age = st.columns(3)

            with c_gen:
                fig_gender = go.Figure()
                fig_gender.add_trace(go.Scatter(
                    y=df_hist["diab_f"], name="Diabetic Female", line_color="#e74c3c", mode="lines+markers"
                ))
                fig_gender.add_trace(go.Scatter(
                    y=df_hist["diab_m"], name="Diabetic Male", line_color="#3498db", mode="lines+markers"
                ))
                fig_gender.update_layout(title="Diabetic Count by Gender", height=280)
                st.plotly_chart(fig_gender, use_container_width=True, key="chart_gender")

            with c_smoke:
                smoke_df = pd.DataFrame({
                    "Smoking": list(stats["smoking_diab"].keys()) + list(stats["smoking_healthy"].keys()),
                    "Diabetic": [stats["smoking_diab"].get(k, 0) for k in stats["smoking_diab"]] + [0] * len(stats["smoking_healthy"]),
                    "Healthy": [0] * len(stats["smoking_diab"]) + [stats["smoking_healthy"].get(k, 0) for k in stats["smoking_healthy"]],
                }).groupby("Smoking").sum().reset_index()

                fig_smoke = px.bar(
                    smoke_df.melt(id_vars="Smoking", var_name="Group", value_name="Count"),
                    x="Smoking", y="Count", color="Group",
                    barmode="group",
                    title="Smoking History by Group (current counts)",
                    color_discrete_sequence=["#e74c3c", "#27ae60"]
                )
                fig_smoke.update_layout(height=280)
                st.plotly_chart(fig_smoke, use_container_width=True, key="chart_smoke")

            with c_age:
                fig_age = go.Figure()
                fig_age.add_trace(go.Histogram(
                    x=s["ages_diab"], name="Diabetic", marker_color="#e74c3c", nbinsx=15, opacity=0.7
                ))
                fig_age.add_trace(go.Histogram(
                    x=s["ages_healthy"], name="Healthy", marker_color="#27ae60", nbinsx=15, opacity=0.7
                ))
                fig_age.update_layout(
                    barmode="overlay",
                    title="Age Distribution (all received data)",
                    xaxis_title="Age",
                    yaxis_title="Count",
                    height=280
                )
                st.plotly_chart(fig_age, use_container_width=True, key="chart_age")

        _live_panel()
import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import STREAM_INTERVAL, STATS_URL
from helpers import stream_controls, http_get

def show_stats_page():
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

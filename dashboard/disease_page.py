import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import STREAM_INTERVAL, DISEASE_URL, Q_DISEASE
from helpers import http_post, consume_one, stream_controls, append_log

def show_disease_page():
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
                            f"🦠 {result['predicted_disease']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric("Model confidence", f"{conf:.1f}%")
                        st.progress(result["confidence"])
                    with c2:
                        st.subheader("📊 Top 3 Candidates")
                        top3 = result.get("top3", [])
                        if top3:
                            names = [t["disease"]          for t in top3]
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
                        "predicted":   result["predicted_disease"],
                        "confidence":  result["confidence"],
                    })
            else:
                st.toast("Queue is empty — waiting for producer...", icon="⏳")
            time.sleep(STREAM_INTERVAL)
            st.rerun()
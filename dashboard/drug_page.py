import streamlit as st
import time
import pandas as pd
import plotly.express as px
from collections import Counter

from config import STREAM_INTERVAL, DRUG_URL, Q_DRUG
from helpers import http_post, consume_one, stream_controls, append_log
from drug_info import display_medicine_card

def show_drug_page():
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
                    
                    if result["drug_list"]:
                        st.divider()
                        st.subheader("📋 معرفی کامل داروها (از دیتابیس)")
                        st.caption("اطلاعات، عوارض و نظرات کاربران")

                        # نمایش کارت‌ها در ستون (حداکثر ۲ در هر ردیف)
                        num_drugs = len(result["drug_list"])
                        cols = st.columns(min(2, num_drugs))

                        for idx, drug in enumerate(result["drug_list"]):
                            with cols[idx % len(cols)]:
                                display_medicine_card(drug)

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

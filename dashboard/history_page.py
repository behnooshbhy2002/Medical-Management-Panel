import streamlit as st
import pandas as pd


from config import API_URL
from helpers import http_get

def show_history_page():
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
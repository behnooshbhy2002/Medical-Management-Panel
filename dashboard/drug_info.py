import sqlite3
import json
import streamlit as st
import plotly.express as px
from config import DB_PATH


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def get_medicine_info(drug_name: str):
    """Smart search: exact match first, then partial match"""
    conn = get_db_connection()
    c = conn.cursor()

    # Try exact match (case-insensitive)
    c.execute("SELECT * FROM medicines WHERE LOWER(name) = LOWER(?)", (drug_name,))
    row = c.fetchone()

    # If not found → try partial match
    if not row:
        c.execute("SELECT * FROM medicines WHERE LOWER(name) LIKE ?", (f"%{drug_name.lower()}%",))
        row = c.fetchone()

    conn.close()

    if row:
        return {
            "name": row[1],
            "composition": row[2],
            "uses": row[3],
            "side_effects": json.loads(row[4]),
            "image_url": row[5],
            "manufacturer": row[6],
            "excellent": row[7],
            "average": row[8],
            "poor": row[9],
        }
    
    return None


def display_medicine_card(drug_name: str):
    """Display a nice-looking medicine info card"""
    info = get_medicine_info(drug_name)
    
    if not info:
        st.warning(f"🚫 No information found for **{drug_name}** in the database.")
        return

    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        
        with c1:
            st.subheader(f"💊 {info['name']}")
            st.caption(f"**Composition:** {info['composition']}")
            st.caption(f"**Manufacturer:** {info['manufacturer']}")
        
        with c2:
            if info["image_url"]:
                st.image(info["image_url"], width=100)

        st.divider()

        st.markdown("**🟢 Uses / Indications:**")
        st.write(info["uses"])

        st.markdown("**🔴 Side Effects:**")
        for effect in info["side_effects"]:
            st.write(f"• {effect}")

        st.markdown("**📊 User Satisfaction Ratings:**")
        fig = px.pie(
            names=["Excellent", "Average", "Poor"],
            values=[info["excellent"], info["average"], info["poor"]],
            color_discrete_sequence=["#27ae60", "#f1c40f", "#e74c3c"],
            hole=0.45
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=260,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)
import streamlit as st
from config import _DEFAULTS

# ── Session state defaults ────────────────────────────────────
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🏥 Health AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Section",
    ["🔬 Diabetes", "🧬 Disease", "💊 Drug", "📊 Stats"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Each page has two tabs:\n"
    "- **Manual**: enter data, get instant result\n"
    "- **Live Stream**: pulls from RabbitMQ every 5 s"
)

# ── Page Routing ──────────────────────────────────────────────
if page == "🔬 Diabetes":
    from diabetes_page import show_diabetes_page
    show_diabetes_page()
elif page == "🧬 Disease":
    from disease_page import show_disease_page
    show_disease_page()
elif page == "💊 Drug":
    from drug_page import show_drug_page
    show_drug_page()
elif page == "📊 Stats":
    from stats_page import show_stats_page
    show_stats_page()

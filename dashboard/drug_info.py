import sqlite3
import json
import streamlit as st
import plotly.express as px
from groq import Groq
import markdown
import re
from config import DB_PATH, GROQ_API_KEY, GROQ_MODEL

# Initialize Groq client (do this once)
@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is not set in config")
    return Groq(api_key=GROQ_API_KEY)


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
    """نمایش کارت دارو – از دیتابیس یا به صورت خودکار از AI"""
    
    card_key = f"card_container_{drug_name.replace(' ', '_')}"
    with st.container(border=True, key=card_key):
        st.subheader(f"💊 {drug_name}")

        info = get_medicine_info(drug_name)

        if info:
            # ── حالت دیتابیس (همان کد قبلی) ──────────────────────────────
            c1, c2 = st.columns([5, 1])
            with c1:
                st.caption(f"**Composition:** {info['composition'] or '—'}")
                st.caption(f"**Manufacturer:** {info['manufacturer'] or '—'}")
            with c2:
                if info.get("image_url"):
                    st.image(info["image_url"], width=100)

            st.divider()
            st.markdown("**🟢 Uses / Indications**")
            st.write(info["uses"] or "اطلاعات موجود نیست")

            st.markdown("**🔴 Common Side Effects**")
            if info["side_effects"]:
                for eff in info["side_effects"]:
                    st.write(f"• {eff}")
            else:
                st.write("—")

            st.markdown("**📊 User Satisfaction**")
            if info["excellent"] + info["average"] + info["poor"] > 0:
                fig = px.pie(
                    names=["Excellent", "Average", "Poor"],
                    values=[info["excellent"], info["average"], info["poor"]],
                    color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=240, margin=dict(t=10, b=10, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("آمار رضایت موجود نیست")

        else:
            placeholder = st.empty()

            with st.spinner("در حال دریافت اطلاعات از هوش مصنوعی..."):
                try:
                    full_response = ask_ai_about_medicine_stream(drug_name, placeholder)

                    def clean_ai_response(text: str) -> str:
                        # Decode escaped HTML entities
                        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

                        # If text already contains HTML tags → remove all tags
                        if re.search(r'<(p|strong|br|div|h[1-6]|ul|li)', text, re.I):
                            text = re.sub(r'<[^>]+>', '', text)
                            text = re.sub(r'\s*\n\s*', '\n', text)

                        # Convert remaining <br> tags to newline
                        text = text.replace('<br>', '\n').replace('<br />', '\n')

                        # 🔹 Remove markdown bold stars (e.g., **Title:** → Title:)
                        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

                        return text.strip()
                    
                    def format_titles(text: str) -> str:
                        text = re.sub(
                            r'^([^:\n]+:)',
                            r'<span style="color:#1f77b4; font-weight:bold;">\1</span>',
                            text,
                            flags=re.MULTILINE
                        )
                        return text
                    
                    clean_text = clean_ai_response(full_response)
                    formatted_text = format_titles(clean_text)
                    # print(clean_text)

                    # حالا تبدیل به html
                    html_content = markdown.markdown(
                        formatted_text,
                        extensions=['extra', 'nl2br', 'sane_lists']
                    )

                    # Inject custom CSS (only once per render is fine)
                    st.markdown("""
                    <style>
                    .ai-card {
                        direction: rtl;
                        text-align: right;
                        font-family: Vazirmatn, IRANSans, Tahoma, sans-serif;
                        background-color: #0f172a;
                        padding: 20px;
                        border-radius: 14px;
                        border: 1px solid #1f2937;
                        line-height: 1.9;
                        font-size: 15px;
                    }

                    .ai-badge {
                        display: inline-block;
                        background: #1e293b;
                        padding: 4px 10px;
                        border-radius: 8px;
                        font-size: 11px;
                        color: #38bdf8;
                        margin-bottom: 12px;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Display styled card
                    st.markdown(f"""
                    <div class="ai-card">
                        <div class="ai-badge">
                            🤖 این محتوا توسط هوش مصنوعی تولید شده است
                        </div>
                        <div>
                        {html_content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"خطا: {str(e)}")

# تابع streaming (تقریباً همان قبلی، اما با placeholder بهتر)
def ask_ai_about_medicine_stream(drug_name: str, placeholder):
    client = get_groq_client()

    prompt = f"""لطفاً اطلاعات دارویی زیر را **فقط به صورت markdown خام** و به زبان فارسی (اگر کلمه ای معادل فارسی نداشت خودش رو بنویسید) ارائه دهید.
    **هیچ تگ HTML استفاده نکنید** (نه <p>، نه <strong>، نه <br> و غیره).
    عوارض جانبی شایع در حد 5 تا 8 مورد
    شرکت‌های سازنده یا برندهای شناخته‌شده اگر وجود داشت بنویس.

    نام دارو: {drug_name}

    ساختار پاسخ (دقیقاً همین فرمت را رعایت کنید):

    **ترکیب اصلی / ماده موثره:**
    ...

    **موارد مصرف اصلی:**
    ...

    **عوارض جانبی شایع:**
    • ...
    • ...

    **هشدارها و موارد احتیاط مهم:**
    ...

    **شرکت‌های سازنده یا برندهای شناخته‌شده:**
    ...

    پاسخ کوتاه، دقیق و فقط جنبه اطلاع‌رسانی داشته باشد. هیچ توصیه پزشکی نکنید."""

    full_response = ""

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # یا "mixtral-8x22b-..." یا مدل مناسب دیگر
        messages=[
            {"role": "system", "content": "شما دستیار اطلاع‌رسانی دارویی هستید."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1200,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            # placeholder.markdown(full_response + " ▌")  # افکت تایپ

    # placeholder.markdown(full_response)
    return full_response
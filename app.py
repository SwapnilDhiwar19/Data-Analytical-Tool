import streamlit as st
import pandas as pd
import json
import os
import time
from google import genai
from datetime import datetime

# --- 1. CONFIGURATION & MODEL SETUP ---
st.set_page_config(
    page_title="Self-Service Analytics | BIU",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Securely retrieve the key from Streamlit Secrets or Environment Variable
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found. Please configure it in your Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-pro"

def generate_content_with_retry(client_obj, prompt_text, max_retries=3):
    """Executes prompt with backoff retries and model failover against 503 capacity spikes."""
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                res = client_obj.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config={"response_mime_type": "application/json"}
                )
                return res
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    wait_seconds = (attempt + 1) * 2
                    time.sleep(wait_seconds)
                    continue
                break
                
    raise Exception("AI servers are currently experiencing high demand. Please try again shortly.")

# --- 2. THEME (Apple Minimalist: SF Pro, subtle grays, soft shadows, accent blue) ---
st.markdown("""
<style>
    /* Global & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
        color: #1d1d1f;
        background-color: #fbfbfd;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 980px;
        background-color: transparent;
    }

    /* Hero / Header Styling */
    .hero-wrap {
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1d1d1f;
        margin-bottom: 0.3rem;
    }
    .hero-tagline {
        text-align: center;
        color: #86868b;
        font-size: 1.1rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        margin-bottom: 0.6rem;
    }
    .hero-privacy-notice {
        display: inline-block;
        background-color: #f5f5f7;
        color: #6e6e73;
        font-size: 0.82rem;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        letter-spacing: -0.01em;
    }

    /* Apple-style File Uploader */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 18px !important;
        border: 1px dashed #d2d2d7 !important;
        background-color: #ffffff !important;
        transition: all 0.2s ease-in-out;
        max-width: 700px;
        margin: 0 auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0071e3 !important;
        background-color: #fbfbfd !important;
    }
    div[data-testid="stFileUploader"] {
        max-width: 700px;
        margin: 0 auto;
    }

    /* Apple-style Buttons */
    .stButton>button {
        background-color: #0071e3;
        color: #ffffff;
        border-radius: 980px;
        border: none;
        font-size: 0.92rem;
        font-weight: 500;
        padding: 0.5rem 1.8rem;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 6px rgba(0, 113, 227, 0.25);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0077ed;
        color: #ffffff;
        transform: scale(1.01);
    }
    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Output Containers */
    .apple-card {
        background: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
        margin-top: 1.5rem;
    }

    .insight-card {
        background: #f5f5f7;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
        color: #1d1d1f;
        line-height: 1.6;
        font-size: 0.94rem;
    }
    .insight-card h4 {
        margin: 0 0 0.4rem 0;
        font-weight: 600;
        font-size: 1rem;
        color: #1d1d1f;
    }

    .history-card {
        background-color: #ffffff;
        border: 1px solid #e5e5ea;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 4. HERO SECTION ---
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Self-Service Analytics</div>
    <div class="hero-tagline">BIU Self-Service Analytics · Automated Cross-tabs, Visuals & Deep Insights</div>
    <div class="hero-privacy-notice">🔒 Only data schema is shared with LLM hence prompt questions accordingly headers</div>
</div>
""", unsafe_allow_html=True)

# --- 5. FILE UPLOAD & EXECUTION ---
uploaded_file = st.file_uploader(" ", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown('<div style="text-align: center; color: #86868b; font-size: 0.88rem; margin-top: 2rem;">Upload a CSV or Excel worksheet to begin</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)

        st.toast(f"Loaded {uploaded_file.name} ({len(df):,} rows)", icon="📊")

        tab_ask, tab_history = st.tabs(["Ask Assistant", "Query History"])

        with tab_ask:
            query = st.text_input("Ask a business question", label_visibility="collapsed", placeholder="Ask a question about your columns or metrics...")
            
            run = False
            _, btn_col, _ = st.columns([2, 1, 2])
            with btn_col:
                run = st.button("Generate Analysis", use_container_width=True)

            if run and query:
                with st.spinner("Analyzing schema and generating insights..."):
                    col_summary = "\n".join([f"- {col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
                    num_summary = df.describe().to_string()

                    prompt = f"""
                    You are a Lead BIU Analytics Consultant.
                    A pandas DataFrame named 'df' is loaded in memory with these columns and types:
                    {col_summary}

                    Summary statistics preview:
                    {num_summary}

                    BUSINESS QUESTION: "{query}"

                    CRITICAL REQUIREMENTS:
                    1. Cross-Tab & Visuals:
                       - Output an aggregated cross-tab or pivot table using `st.dataframe(...)` or `st.table(...)`.
                       - Output a primary visual chart using `st.bar_chart(...)` or `st.line_chart(...)`.
                       - GRAPH REQUIREMENT: Every chart MUST have clear legends, labeled axes, and meaningful multi-column/metric designations so readers know what each color represents. Set index columns intentionally for legends.
                    2. Deep Analytical Writeup:
                       - Write a clear, executive-grade summary referencing numerical trends, ratios, averages, and variances.
                    3. Strict Format:
                       - Return ONLY a valid JSON object without markdown wrapping or backticks.

                    JSON Schema:
                    {{
                        "executive_summary": "Comprehensive 3-5 sentence breakdown referencing numbers, percentages, and direction of trends.",
                        "code": "Valid Python code using Streamlit to compute and display BOTH the cross-tab and visual chart with complete legend readability."
                    }}
                    """

                    response = generate_content_with_retry(client, prompt)
                    
                    try:
                        clean_json = response.text.strip()
                        if clean_json.startswith("```"):
                            clean_json = clean_json.strip("`")
                            if clean_json.lower().startswith("json"):
                                clean_json = clean_json[4:]
                        parsed = json.loads(clean_json)
                        summary = parsed.get("executive_summary", "")
                        clean_code = parsed.get("code", "")
                    except Exception:
                        summary = "Analysis generated. See computations and visuals below."
                        clean_code = response.text.strip().replace("```python", "").replace("```", "")

                    st.session_state.history.append({
                        "query": query,
                        "summary": summary,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })

                    # --- Render Output Inside Result Card ---
                    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
                    st.subheader("Data & Visual Output")
                    
                    # 1. Insights inside the output card
                    if summary:
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>📊 Business Insights & Trends</h4>
                            {summary}
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. Executable cross-tabs & visual charts with legends
                    try:
                        exec(clean_code)
                    except Exception as err:
                        st.error(f"Execution Error: {err}")
                        with st.expander("View generated code"):
                            st.code(clean_code, language="python")

                    st.markdown('</div>', unsafe_allow_html=True)

        with tab_history:
            if not st.session_state.history:
                st.info("No queries executed in this session yet.")
            else:
                for item in reversed(st.session_state.history):
                    st.markdown(f"""
                    <div class="history-card">
                        <strong style="color: #1d1d1f;">🕒 {item['time']} — {item['query']}</strong>
                        <p style="margin: 0.4rem 0 0 0; font-size: 0.9rem; color: #515154;">{item['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")

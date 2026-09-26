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

def generate_content_with_retry(client_obj, prompt_text, response_mime="application/json", max_retries=3):
    """Executes prompt with backoff retries and model failover against capacity spikes."""
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    config = {"response_mime_type": response_mime} if response_mime else {}
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                res = client_obj.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=config
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

# --- 2. THEME (Apple Minimalist with Page Border) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
        color: #1d1d1f;
        background-color: #f5f5f7;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Bounded Apple-style Page Canvas */
    .block-container {
        padding: 3rem 2.5rem;
        max-width: 1000px;
        background-color: #ffffff;
        border: 1px solid #d2d2d7;
        border-radius: 20px;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 28px rgba(0, 0, 0, 0.04);
    }

    /* Hero / Header Styling */
    .hero-wrap {
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1d1d1f;
        margin-bottom: 0.3rem;
    }
    .hero-tagline {
        text-align: center;
        color: #86868b;
        font-size: 1.05rem;
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
        margin-bottom: 1.8rem;
        border: 1px solid #e5e5ea;
        letter-spacing: -0.01em;
    }

    /* Apple-style File Uploader */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
        border: 1px dashed #d2d2d7 !important;
        background-color: #fbfbfd !important;
        transition: all 0.2s ease-in-out;
        max-width: 700px;
        margin: 0 auto;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #0071e3 !important;
        background-color: #ffffff !important;
    }
    div[data-testid="stFileUploader"] {
        max-width: 700px;
        margin: 0 auto;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0071e3;
        color: #ffffff;
        border-radius: 980px;
        border: none;
        font-size: 0.92rem;
        font-weight: 500;
        padding: 0.5rem 1.8rem;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 8px rgba(0, 113, 227, 0.2);
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

    /* Output Card */
    .apple-card {
        background: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.03);
        margin-top: 1.5rem;
    }

    .insight-card {
        background: #f5f5f7;
        border: 1px solid #e5e5ea;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-top: 1.2rem;
        color: #1d1d1f;
        line-height: 1.6;
        font-size: 0.92rem;
    }
    .insight-card h4 {
        margin: 0 0 0.35rem 0;
        font-weight: 600;
        font-size: 0.98rem;
        color: #1d1d1f;
    }

    .history-card {
        background-color: #ffffff;
        border: 1px solid #e5e5ea;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
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
    st.markdown('<div style="text-align: center; color: #86868b; font-size: 0.88rem; margin-top: 1.5rem;">Upload a CSV or Excel worksheet to begin</div>', unsafe_allow_html=True)

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
                with st.spinner("Compiling cross-tab and visual..."):
                    col_summary = "\n".join([f"- {col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])

                    # Step 1: Generate executable code that computes output_df and renders the charts
                    code_prompt = f"""
                    You are a Lead BIU Analytics Engineer.
                    A pandas DataFrame 'df' is loaded in memory with these columns:
                    {col_summary}

                    BUSINESS QUESTION: "{query}"

                    CRITICAL INSTRUCTIONS:
                    1. Compute an aggregated cross-tab or pivot table and ALWAYS assign the resulting DataFrame to a variable named `output_df`.
                    2. Display `output_df` using `st.dataframe(output_df, use_container_width=True)`.
                    3. Create a primary visual chart using `st.bar_chart(...)` or `st.line_chart(...)`.
                       - The chart MUST show clean legends and readable series.
                       - When plotting `output_df`, ensure index or columns are oriented so Streamlit automatically provides clear legends for each category or metric.
                    4. Return ONLY a valid JSON object without markdown formatting or backticks.

                    JSON Schema:
                    {{
                        "code": "Python code assigning results to output_df and rendering the table & chart."
                    }}
                    """

                    code_res = generate_content_with_retry(client, code_prompt)
                    
                    try:
                        clean_json = code_res.text.strip()
                        if clean_json.startswith("```"):
                            clean_json = clean_json.strip("`")
                            if clean_json.lower().startswith("json"):
                                clean_json = clean_json[4:]
                        parsed = json.loads(clean_json)
                        clean_code = parsed.get("code", "")
                    except Exception:
                        clean_code = code_res.text.strip().replace("```python", "").replace("```", "")

                    # Render Output Canvas
                    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
                    st.subheader("Data & Visual Output")
                    
                    execution_scope = {"df": df, "st": st, "pd": pd}
                    exec_error = None
                    try:
                        exec(clean_code, execution_scope)
                    except Exception as err:
                        exec_error = err
                        st.error(f"Execution Error: {err}")
                        with st.expander("View generated code"):
                            st.code(clean_code, language="python")

                    # Step 2: Generate Insights STRICTLY on the computed output
                    output_df = execution_scope.get("output_df")
                    summary = ""

                    if exec_error is None and output_df is not None:
                        with st.spinner("Synthesizing insights strictly from output results..."):
                            output_preview = output_df.to_string() if hasattr(output_df, "to_string") else str(output_df)

                            insight_prompt = f"""
                            You are a Senior MIS & Business Intelligence Analyst.
                            The user asked: "{query}"

                            Below is the ACTUAL aggregated data output produced by the query:
                            {output_preview}

                            TASK:
                            Write an executive analytical writeup of 3 to 4 concise sentences.
                            
                            STRICT RULES:
                            - Speak ONLY about the numbers, percentages, ranks, maximums, minimums, and deltas present in the output above.
                            - Do NOT assume, speculate, or introduce external context not represented in this exact output.
                            - Format directly as clear, plain narrative text without bullet points or headers.
                            """

                            insight_res = generate_content_with_retry(client, insight_prompt, response_mime=None)
                            summary = insight_res.text.strip()

                            st.markdown(f"""
                            <div class="insight-card">
                                <h4>📊 Business Insights & Trends</h4>
                                {summary}
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Update history
                    if summary:
                        st.session_state.history.append({
                            "query": query,
                            "summary": summary,
                            "time": datetime.now().strftime("%H:%M:%S")
                        })

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

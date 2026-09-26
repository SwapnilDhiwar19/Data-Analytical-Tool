import os
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Data Analytical Tool (BIU)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# 2. Apple Light-Mode CSS Injection
# ---------------------------------------------------------
apple_light_css = """
<style>
@import url('https://fonts.cdnfonts.com/css/sf-pro-display');

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
    background-color: #fbfbfd !important;
    color: #1d1d1f !important;
    letter-spacing: -0.012em;
}

/* Hide default Streamlit chrome */
#MainMenu, header, footer, [data-testid="stDecoration"] {
    visibility: hidden;
    height: 0%;
}

/* Layout container adjustments */
.block-container {
    max-width: 1040px;
    padding-top: 3.5rem;
    padding-bottom: 5rem;
}

/* Surface Cards */
div[data-testid="stMetric"], .apple-card {
    background: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.07) !important;
    border-radius: 20px !important;
    padding: 24px 28px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

div[data-testid="stMetric"]:hover, .apple-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.07), 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}

/* Metric styling */
div[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #86868b !important;
}

div[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.025em !important;
}

/* Typography */
h1 {
    font-weight: 700 !important;
    font-size: 3.4rem !important;
    letter-spacing: -0.035em !important;
    text-align: center;
    background: linear-gradient(180deg, #1d1d1f 0%, #48484a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #1d1d1f !important;
}

/* Apple Pill Action Buttons */
div.stButton > button {
    background: #0071e3 !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border-radius: 980px !important;
    border: none !important;
    padding: 10px 24px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(0, 113, 227, 0.25);
}

div.stButton > button:hover {
    background: #0077ed !important;
    transform: scale(1.015);
    box-shadow: 0 4px 14px rgba(0, 113, 227, 0.35);
}

div.stButton > button:active {
    transform: scale(0.98);
}

/* Minimal Input Styling */
div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #d2d2d7 !important;
    border-radius: 12px !important;
    color: #1d1d1f !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03);
    transition: border-color 0.2s ease;
}

div[data-baseweb="input"]:focus-within {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15) !important;
}

input::placeholder {
    color: #86868b !important;
}
</style>
"""
st.markdown(apple_light_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. API Key Resolution
# ---------------------------------------------------------
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif os.environ.get("GEMINI_API_KEY"):
    api_key = os.environ.get("GEMINI_API_KEY")

# Fallback: Allow key entry via sidebar if not configured in secrets
if not api_key:
    with st.sidebar:
        st.subheader("Configuration")
        api_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Get your key at https://aistudio.google.com/",
        )

# ---------------------------------------------------------
# 4. Hero Section
# ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 40px;">
        <span style="font-size: 0.85rem; font-weight: 600; color: #0071e3; letter-spacing: 0.08em; text-transform: uppercase;">
            Intelligence System
        </span>
        <h1>Engineered for capital.</h1>
        <p style="font-size: 1.25rem; color: #6e6e73; max-width: 580px; margin: 8px auto 0 auto; line-height: 1.5;">
            Institutional-grade valuation metrics with zero distraction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 5. Search Interface
# ---------------------------------------------------------
_, col_search, _ = st.columns([1, 2, 1])
with col_search:
    ticker = st.text_input(
        "Ticker",
        placeholder="Enter Ticker Symbol (e.g. AAPL, NVDA, MSFT)",
        label_visibility="collapsed",
    )
    analyze_btn = st.button("Generate Intelligence", use_container_width=True)

# ---------------------------------------------------------
# 6. Analysis Execution
# ---------------------------------------------------------
if analyze_btn:
    if not api_key:
        st.error("Please configure `GEMINI_API_KEY` in Streamlit secrets or enter it in the sidebar.")
    elif not ticker.strip():
        st.warning("Please enter a valid stock ticker symbol.")
    else:
        with st.spinner("Compounding intelligence..."):
            try:
                client = genai.Client(api_key=api_key)

                prompt = f"""
                You are a senior equity research analyst. Analyze the company with ticker: {ticker.strip().upper()}.

                Provide:
                1. A brief 2-sentence summary of the business and competitive moat.
                2. Three primary growth and earnings drivers.
                3. Capital allocation track record (ROIC, reinvestment vs returns).

                CRITICAL INSTRUCTION:
                Do NOT include any 'Analytical Caveats & Distribution Risks', caveats, disclaimer, or risk warnings. Provide strictly the direct fundamental analysis.
                """

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                # Clean text defense against generic disclaimer triggers
                analysis_text = response.text
                for forbidden_header in [
                    "⚠️ Analytical Caveats & Distribution Risks",
                    "Analytical Caveats & Distribution Risks",
                    "Analytical Caveats",
                    "Distribution Risks",
                ]:
                    if forbidden_header in analysis_text:
                        analysis_text = analysis_text.split(forbidden_header)[0]

                # Metric Snapshot (clean Apple cards)
                st.write("")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="Target Symbol", value=ticker.strip().upper(), delta="Active")
                with m2:
                    st.metric(label="Model Engine", value="Gemini 2.5", delta="Low Latency")
                with m3:
                    st.metric(label="Analysis Focus", value="Fundamental", delta="Core Equity")

                # Content Card
                st.markdown(
                    f"""
                    <div class="apple-card" style="margin-top: 24px;">
                        <span style="font-size: 0.8rem; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em;">
                            Equity Thesis
                        </span>
                        <h3 style="margin-top: 8px; margin-bottom: 16px;">{ticker.strip().upper()} Overview & Drivers</h3>
                        <div style="color: #333336; font-size: 1.05rem; line-height: 1.65;">
                            {analysis_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

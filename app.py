import os
import time
import ast
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- 1. CONFIGURATION & MODEL SETUP ---
st.set_page_config(
    page_title="AI Driven Self-serviced Analytics | BIU",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY not configured. Set it in `.streamlit/secrets.toml` or environment variables.")
    st.stop()

@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

client = get_gemini_client(GEMINI_API_KEY)

PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-pro"

# --- 2. STRUCTURED RESPONSE SCHEMAS ---
class CodeGenerationResponse(BaseModel):
    code: str = Field(description="Pure Python code defining 'res_df' (aggregated DataFrame) and optionally 'chart_type' ('bar' or 'line').")
    chart_type: str = Field(default="bar", description="One of 'bar' or 'line'.")
    x_axis: str = Field(description="Column name in res_df to use as the index / x-axis.")
    y_axis: list[str] = Field(description="List of numeric columns in res_df to plot on the y-axis.")

class AnalyticalReportResponse(BaseModel):
    executive_summary: str = Field(description="3-5 bulleted or prose analytical sentences referencing verified numbers from the aggregate result.")
    caveats_and_risks: str = Field(description="2-3 sentences discussing sample distribution, concentrations, or missing slices.")

# --- 3. SAFE EXECUTION GUARDRAIL ---
BANNED_AST_NODES = (
    ast.Import, ast.ImportFrom,
    ast.Global, ast.Nonlocal,
    ast.AsyncFunctionDef, ast.ClassDef
)

BANNED_NAMES = {
    "open", "exec", "eval", "compile", "globals", "locals", "__import__",
    "os", "sys", "subprocess", "socket", "requests", "shutil", "builtins",
    "pathlib", "ctypes", "pickle"
}

def validate_code_safety(code_str: str) -> None:
    """Verifies that the generated code contains no dangerous constructs or imports."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as err:
        raise ValueError(f"Generated code has syntax errors: {err}")

    for node in ast.walk(tree):
        if isinstance(node, BANNED_AST_NODES):
            raise SecurityError(f"Security Alert: Disallowed operation '{type(node).__name__}' detected.")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SecurityError(f"Security Alert: Access to dangerous variable or module '{node.id}' blocked.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SecurityError("Security Alert: Direct dunder access is strictly prohibited.")

def execute_sandboxed_transformation(code_str: str, source_df: pd.DataFrame) -> dict:
    """Executes validated code in a restricted scope and guarantees an output DataFrame."""
    validate_code_safety(code_str)

    safe_globals = {
        "__builtins__": {
            "range": range, "len": len, "min": min, "max": max,
            "sum": sum, "abs": abs, "round": round, "enumerate": enumerate,
            "zip": zip, "int": int, "float": float, "str": str, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "bool": bool
        },
        "pd": pd,
        "np": np
    }
    safe_locals = {"df": source_df.copy()}

    exec(code_str, safe_globals, safe_locals)

    if "res_df" not in safe_locals or not isinstance(safe_locals["res_df"], pd.DataFrame):
        raise ValueError("The generated script failed to assign an aggregated pandas DataFrame to `res_df`.")

    return safe_locals

# --- 4. MODEL CALL UTILITY ---
def call_model_with_retry(prompt: str, schema, max_retries: int = 3):
    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    for model_name in models:
        for attempt in range(max_retries):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1
                    )
                )
                return schema.model_validate_json(res.text)
            except Exception as e:
                err_str = str(e)
                if any(token in err_str for token in ["503", "UNAVAILABLE", "429"]):
                    time.sleep((attempt + 1) * 2)
                    continue
                break
    raise RuntimeError("Upstream Gemini services are unavailable. Please retry shortly.")

# --- 5. DATA INGESTION & METADATA PREP ---
@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename: str) -> pd.DataFrame:
    file_ext = filename.split(".")[-1].lower()
    if file_ext == "csv":
        return pd.read_csv(file_bytes)
    return pd.read_excel(file_bytes)

def extract_dataframe_metadata(input_df: pd.DataFrame) -> str:
    metadata_lines = []
    for col in input_df.columns:
        dtype = str(input_df[col].dtype)
        num_nulls = int(input_df[col].isna().sum())
        
        if pd.api.types.is_numeric_dtype(input_df[col]):
            desc = f"- **{col}** ({dtype}): nulls={num_nulls}, min={input_df[col].min()}, max={input_df[col].max()}"
        else:
            unique_samples = [str(x) for x in input_df[col].dropna().unique()[:5]]
            desc = f"- **{col}** ({dtype}): nulls={num_nulls}, unique_count={input_df[col].nunique()}, samples={unique_samples}"
        metadata_lines.append(desc)
    return "\n".join(metadata_lines)

# --- 6. STYLING (HDFC PALETTE) ---
HDFC_BLUE = "#004C8F"
HDFC_RED = "#ED232A"

st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{
        padding: 2.5rem 3rem;
        max-width: 1000px;
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(0,76,143,0.06);
    }}
    .logo-text {{
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-align: center;
        margin-bottom: 0.2rem;
    }}
    .logo-text .b {{ color: {HDFC_BLUE}; }}
    .logo-text .r {{ color: {HDFC_RED}; }}
    .writeup-box {{
        background-color: #f7faff;
        border-left: 4px solid {HDFC_BLUE};
        border-radius: 6px;
        padding: 1.1rem;
        margin: 1rem 0;
        font-size: 0.95rem;
    }}
    .caveat-box {{
        background-color: #fff8f8;
        border-left: 4px solid {HDFC_RED};
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
    }}
</style>
<div class="logo-text"><span class="b">Investor Analysis </span><span class="r">AI-Engine</span></div>
<p style="text-align:center; color:#5f6368; font-size:0.9rem; margin-bottom:1.5rem;">Verified BIU Self-Service Intelligence Engine</p>
""", unsafe_allow_html=True)

# --- 7. SESSION STATE INITIALIZATION ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# --- 8. RUNTIME WORKFLOW ---
uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file, uploaded_file.name)
        st.success(f"Loaded **{uploaded_file.name}** ({len(df):,} rows × {len(df.columns)} columns)")

        tab_ask, tab_history = st.tabs(["Analyze", "History"])

        with tab_ask:
            query = st.text_input("Ask a business or credit question:", placeholder="e.g., Cross-tabulate disbursement total and average delinquency across zones")
            
            if st.button("Generate Verified Analysis", type="primary") and query:
                with st.status("Executing rigorous multi-step analysis...", expanded=True) as status:
                    # Step 1: Prepare column schema
                    st.write("Extracting dimensional metadata...")
                    schema_summary = extract_dataframe_metadata(df)

                    # Step 2: Request computation code only
                    st.write("Synthesizing deterministic transformation code...")
                    code_prompt = f"""
                    You are a lead quantitative analyst. Generate pure Python code using pandas that aggregates data to answer:
                    "{query}"

                    Columns & Data Profile:
                    {schema_summary}

                    RULES:
                    1. Input DataFrame is 'df'. Output aggregated DataFrame MUST be assigned to variable 'res_df'.
                    2. Handle NaNs/null values gracefully (e.g., fillna or dropna where appropriate).
                    3. Do not plot or import matplotlib/streamlit. Return only the data transformation code.
                    4. Identify x_axis and y_axis for plotting.
                    """
                    code_res: CodeGenerationResponse = call_model_with_retry(code_prompt, CodeGenerationResponse)

                    # Step 3: Run the code safely
                    st.write("Executing aggregation locally...")
                    exec_env = execute_sandboxed_transformation(code_res.code, df)
                    res_df = exec_env["res_df"]

                    # Step 4: Pass actual results back to LLM for verified insights
                    st.write("Deriving facts from aggregated outputs...")
                    actual_table_str = res_df.head(25).to_markdown()
                    analysis_prompt = f"""
                    You are a Lead Credit BIU Analyst. Write insights based ONLY on this verified aggregated data:
                    
                    USER QUERY: "{query}"
                    
                    AGGREGATED RESULT:
                    {actual_table_str}

                    RULES:
                    - You must quote exact metrics, trends, and rate calculations from the aggregate table above.
                    - Highlight significant outliers, top drivers, and distribution risks based on this calculated result.
                    """
                    report_res: AnalyticalReportResponse = call_model_with_retry(analysis_prompt, AnalyticalReportResponse)
                    status.update(label="Analysis complete!", state="complete", expanded=False)

                    # Persist state
                    st.session_state.current_analysis = {
                        "query": query,
                        "res_df": res_df,
                        "chart_type": code_res.chart_type,
                        "x_axis": code_res.x_axis,
                        "y_axis": code_res.y_axis,
                        "summary": report_res.executive_summary,
                        "caveats": report_res.caveats_and_risks,
                        "code": code_res.code,
                        "time": datetime.now().strftime("%H:%M:%S")
                    }

                    st.session_state.history.append({
                        "query": query,
                        "summary": report_res.executive_summary,
                        "time": datetime.now().strftime("%H:%M:%S")
                    })

            # Render output components
            analysis = st.session_state.current_analysis
            if analysis:
                # 1. Executive Summary & Core Insights
                st.markdown(f"""
                <div class="writeup-box">
                    <h4 style="margin:0 0 0.5rem 0; color:{HDFC_BLUE};">📊 Business Insights & Analysis</h4>
                    {analysis["summary"]}
                </div>
                """, unsafe_allow_html=True)

                # 2. Aggregated Tabular Cross-Tab
                st.subheader("Aggregated Cross-Tab")
                st.dataframe(analysis["res_df"], use_container_width=True)

                # 3. Primary Visual Output
                st.subheader("Data Visualization")
                try:
                    chart_df = analysis["res_df"].copy()
                    x_col = analysis["x_axis"]
                    y_cols = [c for c in analysis["y_axis"] if c in chart_df.columns]

                    if x_col in chart_df.columns:
                        chart_df = chart_df.set_index(x_col)
                    
                    if y_cols:
                        chart_data = chart_df[y_cols]
                        if analysis["chart_type"] == "line":
                            st.line_chart(chart_data)
                        else:
                            st.bar_chart(chart_data)
                    else:
                        st.bar_chart(chart_df.select_dtypes(include=[np.number]))
                except Exception as chart_err:
                    st.warning(f"Visualization fallback: {chart_err}")
                    st.bar_chart(analysis["res_df"].select_dtypes(include=[np.number]))

                # 4. Caveats & Analytical Distribution Risks (Rendered strictly at output conclusion)
                if analysis.get("caveats"):
                    st.markdown(f"""
                    <div class="caveat-box">
                        <strong>⚠️ Analytical Caveats & Distribution Risks:</strong><br>
                        {analysis["caveats"]}
                    </div>
                    """, unsafe_allow_html=True)

                # 5. Technical Audit / Code View
                with st.expander("Inspect Generated Code"):
                    st.code(analysis["code"], language="python")

        with tab_history:
            if not st.session_state.history:
                st.info("No prior queries recorded in this session.")
            else:
                for item in reversed(st.session_state.history):
                    st.markdown(f"**🕒 {item['time']} — {item['query']}**")
                    st.write(item["summary"])
                    st.divider()

    except Exception as e:
        st.error(f"Application Error: {e}")

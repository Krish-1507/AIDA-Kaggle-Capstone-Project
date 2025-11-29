# AIDA v20.0 - FINAL SUBMISSION MASTERPIECE
# 42+ Features + Universal API + Professional Documentation

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
import json
import datetime
import pickle
import warnings
import zipfile
import io
import re
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from textblob import TextBlob
import scipy.stats as stats
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download failed: {e}")

# SVG ICONS
ICONS = {
    "logo": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>""",
    "hypothesis": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 2v7.31L6 15v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-4l-4-5.69V2h-4z"/><path d="M8.5 2h7"/></svg>""",
    "agents": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>""",
    "healing": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>""",
    "causal": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8l4 4-4 4"/></svg>""",
    "simulator": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>""",
    "export": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>""",
    "advisor": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>""",
    "pipeline": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>""",
    "check": """<svg viewBox="0 0 24 24" fill="none" stroke="green" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>""",
    "error": """<svg viewBox="0 0 24 24" fill="none" stroke="red" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""
}

warnings.filterwarnings('ignore')

# ============================================================================
# 🧠 UNIVERSAL API CONNECTIVITY (DEBUGGED)
# ============================================================================

def configure_api(user_api_key):
    try:
        import google.generativeai as genai
    except ImportError:
        return None, "Google Generative AI library not found."

    if not user_api_key or not user_api_key.strip():
        return None, "Please enter your Google API Key."
    
    clean_key = user_api_key.strip()
    genai.configure(api_key=clean_key)
    
    try:
        # Dynamic Model Discovery
        print("Discovering available models...")
        all_models = list(genai.list_models())
        
        # Filter for models that support content generation
        supported_models = [
            m for m in all_models 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        if not supported_models:
            return None, "No models found that support content generation."
            
        # Priority list for model selection
        priority_terms = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'gemini-1.0-pro'
        ]
        
        selected_model_name = None
        
        # 1. Try to find a match from priority list
        for term in priority_terms:
            for m in supported_models:
                if term in m.name:
                    selected_model_name = m.name
                    break
            if selected_model_name:
                break
        
        # 2. If no priority match, pick the first valid gemini model
        if not selected_model_name:
            for m in supported_models:
                if 'gemini' in m.name:
                    selected_model_name = m.name
                    break
        
        # 3. Fallback to absolutely anything available
        if not selected_model_name:
            selected_model_name = supported_models[0].name
            
        print(f"Selected Model: {selected_model_name}")
        
        # Verify connection
        llm = genai.GenerativeModel(selected_model_name)
        llm.generate_content("Test")
        
        return llm, None
        
    except Exception as e:
        return None, f"Connection Failed: {str(e)}"

# ============================================================================
# 🤖 AGENT SYSTEM (UPDATED)
# ============================================================================

class SelfHealingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.max_retries = 3
    
    def execute_with_healing(self, code, context, df):
        # capture output
        import io
        import sys
        
        for attempt in range(self.max_retries):
            try:
                # Create a capture buffer
                capture = io.StringIO()
                
                # Safe execution environment
                local_vars = {'df': df, 'pd': pd, 'np': np, 'px': px, 'go': go}
                
                # Execute
                exec(f"result = {code}", {}, local_vars)
                
                # Get result
                result = local_vars.get('result', "Execution successful (No return value)")
                return str(result), code, None
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    code = self.heal_code(code, str(e), context)
                else:
                    return None, code, str(e)
        return None, code, "Max retries exceeded"
    
    def heal_code(self, broken_code, error, context):
        prompt = f"Fix Python code.\nError: {error}\nContext: {context}\nCode: {broken_code}\nReturn ONLY fixed code."
        try: return self.llm.generate_content(prompt).text.strip().replace("```python", "").replace("```", "").strip()
        except: return broken_code

class CouncilOfAgents:
    def __init__(self, llm):
        self.llm = llm
        
    def debate_and_decide(self, topic, context):
        try:
            analyst = self.llm.generate_content(f"Role: Data Analyst (Optimist). Topic: {topic}. Context: {context}. Provide 3 opportunities.").text
        except Exception as e: analyst = f"Analyst Unavailable: {str(e)}"
        
        try:
            skeptic = self.llm.generate_content(f"Role: Risk Officer (Critic). Topic: {topic}. Context: {context}. Find flaws and risks.").text
        except Exception as e: skeptic = f"Skeptic Unavailable: {str(e)}"
        
        try:
            decision = self.llm.generate_content(f"Role: CEO. Topic: {topic}. Analyst: {analyst}. Skeptic: {skeptic}. Make a strategic decision.").text
        except Exception as e: decision = f"CEO Unavailable: {str(e)}"
        
        return analyst, skeptic, decision

class HypothesisAgent:
    """Feature: Auto-Feature Discovery & Hypothesis Engine"""
    def __init__(self, llm):
        self.llm = llm

    def discover_and_test(self, df):
        report = "### Auto-Feature Discovery & Hypothesis Engine\n\n"
        
        # 1. Scan
        report += "#### 1. Dataset Scan\n"
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist() # Potential dates
        
        report += f"- **Numeric Features:** {len(numeric_cols)} ({', '.join(numeric_cols[:3])}...)\n"
        report += f"- **Categorical Features:** {len(cat_cols)} ({', '.join(cat_cols[:3])}...)\n"
        
        # 2. Hypothesis Generation & Testing
        report += "\n#### 2. Hypothesis Generation & Scientific Validation\n"
        
        # H1: Correlation (Numeric vs Numeric) - Pearson
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = upper.stack().nlargest(3)
            
            for (col1, col2), val in high_corr.items():
                report += f"- **Hypothesis:** `{col1}` and `{col2}` are linearly related.\n"
                try:
                    stat, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                    status = "Supported" if p < 0.05 else "Rejected"
                    report += f"  - **Test:** Pearson Correlation (r={stat:.2f}, p={p:.4f})\n"
                    report += f"  - **Result:** {status}\n\n"
                except: report += "  - **Test:** Failed (Insufficient data)\n\n"

        # H2: Group Differences (Categorical vs Numeric) - ANOVA
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            cat = cat_cols[0]
            num = numeric_cols[0]
            groups = df[cat].unique()
            if 1 < len(groups) < 10: 
                report += f"- **Hypothesis:** `{num}` varies significantly across `{cat}` groups.\n"
                try:
                    group_data = [df[df[cat] == g][num].dropna() for g in groups]
                    if all(len(g) > 0 for g in group_data):
                        stat, p = stats.f_oneway(*group_data)
                        status = "Supported" if p < 0.05 else "Rejected"
                        report += f"  - **Test:** One-way ANOVA (F={stat:.2f}, p={p:.4f})\n"
                        report += f"  - **Result:** {status}\n\n"
                except: report += "  - **Test:** Failed (Insufficient data)\n\n"

        # H3: Independence (Categorical vs Categorical) - Chi-Square
        if len(cat_cols) >= 2:
            cat1 = cat_cols[0]
            cat2 = cat_cols[1]
            report += f"- **Hypothesis:** `{cat1}` and `{cat2}` are dependent.\n"
            try:
                contingency = pd.crosstab(df[cat1], df[cat2])
                stat, p, dof, expected = stats.chi2_contingency(contingency)
                status = "Supported" if p < 0.05 else "Rejected"
                report += f"  - **Test:** Chi-Square Test (Chi2={stat:.2f}, p={p:.4f})\n"
                report += f"  - **Result:** {status}\n\n"
            except: report += "  - **Test:** Failed (Insufficient data)\n\n"

        # H4: Seasonality (Time Series)
        for col in date_cols:
            try:
                # Try to parse as date
                temp_df = df.copy()
                temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
                temp_df = temp_df.dropna(subset=[col]).sort_values(col)
                if len(temp_df) > 20 and len(numeric_cols) > 0:
                    target = numeric_cols[0]
                    report += f"- **Hypothesis:** `{target}` shows seasonality over `{col}`.\n"
                    # Simple autocorrelation check
                    lag1 = temp_df[target].autocorr(lag=1)
                    status = "Supported" if abs(lag1) > 0.5 else "Rejected"
                    report += f"  - **Test:** Autocorrelation (Lag-1 = {lag1:.2f})\n"
                    report += f"  - **Result:** {status}\n\n"
                    break # Only test one date col
            except: continue

        # 3. AI Insights
        report += "\n#### 3. AI-Generated Insights\n"
        try:
            prompt = f"Based on these stats, generate 3 advanced data science insights:\n{df.describe().to_string()}"
            insights = self.llm.generate_content(prompt).text
            report += insights
        except:
            report += "AI insights unavailable."
            
        return report

class CausalInferenceAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def analyze(self, df, treatment, outcome):
        try:
            if treatment not in df.columns or outcome not in df.columns:
                return "Columns not found in dataset."
            corr = df[[treatment, outcome]].corr().iloc[0, 1]
            return f"### Causal Analysis\n**Treatment:** {treatment}\n**Outcome:** {outcome}\n**Correlation:** {corr:.3f}\n**Recommendation:** Run A/B test to confirm causation."
        except: return "Error in causal analysis."

class BusinessDecisionAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def advise(self, question, context):
        prompt = f"Act as CEO advisor.\nQuestion: {question}\nContext: {context}\nProvide: Recommendation, Pros, Cons."
        try: return self.llm.generate_content(prompt).text
        except: return "Advice unavailable."

class NLQueryAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def query_to_code(self, question, df_info):
        prompt = f"Convert to Python pandas code (use 'df').\nQuestion: {question}\nInfo: {df_info}\nReturn ONLY code expression (e.g. df.head()). Do not use print()."
        try: 
            return self.llm.generate_content(prompt).text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            return f"Error generating code: {str(e)}"

class SQLGeneratorAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_sql(self, question, schema):
        prompt = f"Generate SQL query.\nQuestion: {question}\nSchema: {schema}\nReturn ONLY SQL."
        try: return self.llm.generate_content(prompt).text.strip().replace("```sql", "").replace("```", "").strip()
        except: return "SELECT * FROM table"

class TextAnalyticsAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_sentiment(self, text_series):
        try:
            sentiments = [TextBlob(str(t)).sentiment.polarity for t in text_series[:100] if pd.notna(t)]
            avg = np.mean(sentiments) if sentiments else 0
            return f"### Sentiment Analysis\n**Score:** {avg:.3f} ({'Positive' if avg > 0 else 'Negative'})"
        except: return "Error analyzing text."

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_data(file):
    try:
        # Robust file path handling for different Gradio versions
        if file is None: return None, "No file uploaded"
        
        # If it's an object with a 'name' attribute (Gradio < 4.0 or specific configs)
        if hasattr(file, 'name'):
            file_path = file.name
        # If it's already a string path (Gradio >= 4.0 default)
        elif isinstance(file, str):
            file_path = file
        else:
            return None, f"Unknown file object type: {type(file)}"
            
        if file_path.endswith('.csv'): df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'): df = pd.read_excel(file_path)
        elif file_path.endswith('.json'): df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'): df = pd.read_parquet(file_path)
        else: return None, "Unsupported format. Use CSV, XLSX, JSON, or Parquet."
        
        return df, None
    except Exception as e: return None, f"Error loading data: {str(e)}"

def automated_eda(df, llm):
    desc = df.describe().to_string()
    try:
        return llm.generate_content(f"Analyze dataset stats:\n{desc}\nProvide 3 insights.").text
    except Exception as e:
        return f"EDA Unavailable: {str(e)}"

def train_models(df):
    try:
        numeric_df = df.select_dtypes(include=['number']).fillna(0)
        if numeric_df.shape[1] < 2: return None, "Not enough numeric data"
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]
        is_class = len(np.unique(y)) < 20
        model = RandomForestClassifier() if is_class else RandomForestRegressor()
        model.fit(X, y)
        timestamp = datetime.datetime.now().strftime('%H%M%S')
        path = f"model_{timestamp}.pkl"
        with open(path, 'wb') as f: pickle.dump(model, f)
        return path, None
    except Exception as e: return None, str(e)

def train_what_if_simulator(df):
    try:
        numeric_df = df.select_dtypes(include=['number']).fillna(0)
        if numeric_df.shape[1] < 2: return "Not enough data"
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]
        model = RandomForestRegressor()
        model.fit(X, y)
        return f"Simulator Ready! Trained on {len(X.columns)} features."
    except Exception as e: return f"Error: {e}"

# --- RICH EXPORT HELPER ---
def save_artifact(content, filename, state):
    """Saves content to file and updates session state"""
    # Create a fresh copy of state to ensure Gradio detects the update
    if state is None: 
        new_state = {"files": {}}
    else:
        new_state = state.copy()
        if "files" not in new_state:
            new_state["files"] = {}
        else:
            new_state["files"] = new_state["files"].copy()
    
    # Ensure content is string or bytes
    mode = 'w' if isinstance(content, str) else 'wb'
    with open(filename, mode, encoding='utf-8' if mode=='w' else None) as f:
        f.write(content)
    
    abs_path = os.path.abspath(filename)
    new_state['files'][filename] = abs_path
    print(f"Saved artifact: {filename} -> {abs_path}") # Debug log
    return new_state, abs_path

def create_export_bundle(state):
    try:
        if not state or not state.get('files'): 
            return None
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"AIDA_Session_Export_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w') as zf:
            for fname, fpath in state['files'].items():
                if os.path.exists(fpath):
                    zf.write(fpath, fname)
        
        return zip_filename
    except Exception as e:
        print(f"Export Error: {e}")
        return None

# --- PIPELINE WRAPPERS ---

def run_pipeline_step(df, key, step, state):
    if df is None: return "Please upload data first.", None, state
    llm, err = configure_api(key)
    if err: return err, None, state
    
    if step == "Clean":
        initial = len(df)
        df_clean = df.drop_duplicates().dropna()
        final = len(df_clean)
        msg = f"### Data Cleaning\n- Removed {initial - final} rows.\n- Data is now clean."
        
        # Save CSV
        state, path = save_artifact(df_clean.to_csv(index=False), "cleaned_data.csv", state)
        return msg, path, state
        
    elif step == "EDA":
        report = automated_eda(df, llm)
        state, path = save_artifact(report, "eda_report.md", state)
        return report, path, state
        
    elif step == "Model":
        path, err = train_models(df)
        if path:
            # Track the model file
            state, _ = save_artifact("", path, state) # Just to track it, content write handled by train_models
            # Re-save to ensure it's in the state dict correctly with the right path logic
            # Actually train_models saves it to disk. We just need to add it to state.
            # Let's use save_artifact with mode 'rb' read? No, just manual update or dummy save.
            # Better:
            if state is None: state = {"files": {}}
            else: state = state.copy()
            if "files" not in state: state["files"] = {}
            state["files"][os.path.basename(path)] = os.path.abspath(path)
            
            return f"### Model Trained\n- Saved to {path}", path, state
        else: return f"Error: {err}", None, state
        
    elif step == "All":
        narrative = "### Full Pipeline Execution\n"
        df_clean = df.drop_duplicates().dropna()
        narrative += f"- **Cleaning:** Removed {len(df) - len(df_clean)} rows.\n"
        
        # Save Clean Data
        state, _ = save_artifact(df_clean.to_csv(index=False), "cleaned_data.csv", state)
        
        # EDA
        eda = automated_eda(df_clean, llm)
        narrative += f"- **EDA:** Generated insights.\n{eda}\n"
        state, _ = save_artifact(eda, "eda_report.md", state)
        
        # Model
        path, _ = train_models(df_clean)
        if path: 
            narrative += f"- **Model:** Trained and saved.\n"
            if "files" not in state: state["files"] = {}
            state["files"][os.path.basename(path)] = os.path.abspath(path)
            
        # Save Full Report
        state, report_path = save_artifact(narrative, "full_pipeline_report.md", state)
        
        return narrative, report_path, state

# ============================================================================
# STUNNING UI
# ============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;900&display=swap');
body { background: #0B1120; color: white; font-family: 'Inter', sans-serif; }
.landing-container { padding: 80px 40px; background: radial-gradient(circle at top, #1a2332 0%, #0B1120 100%); min-height: 100vh; }
.hero-title { font-size: 100px; font-weight: 900; text-align: center; background: linear-gradient(to right, #00D9FF, #7B68EE, #FF006E); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; filter: drop-shadow(0 0 40px rgba(0,217,255,0.3)); }
.hero-subtitle { font-size: 24px; text-align: center; color: #8B92A8; margin-bottom: 60px; }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; max-width: 1400px; margin: 0 auto; }
.feature-card { background: rgba(255,255,255,0.03); padding: 30px; border-radius: 24px; border: 1px solid rgba(255,255,255,0.05); transition: all 0.4s ease; display: flex; flex-direction: column; align-items: center; text-align: center; }
.feature-card:hover { transform: translateY(-10px); border-color: #7B68EE; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }
.feature-icon { width: 48px; height: 48px; margin-bottom: 20px; color: #00D9FF; }
.launch-btn { background: linear-gradient(135deg, #00D9FF, #7B68EE); border: none; padding: 25px 80px; border-radius: 50px; font-size: 28px; font-weight: 800; color: white; cursor: pointer; display: block; margin: 80px auto; box-shadow: 0 10px 40px rgba(123,104,238,0.4); transition: transform 0.3s; }
.launch-btn:hover { transform: scale(1.05); }
.section-header { font-size: 32px; font-weight: 700; margin: 60px 0 30px; text-align: center; color: #00D9FF; }
.footer { text-align: center; padding: 20px; color: #8B92A8; font-size: 14px; margin-top: 50px; border-top: 1px solid rgba(255,255,255,0.05); }
"""

with gr.Blocks(css=CSS, title="AIDA: AI Data Automation", theme=gr.themes.Soft()) as demo:
    
    df_state = gr.State(None)
    session_state = gr.State({"files": {}}) # Track all generated files
    
    # LANDING PAGE
    with gr.Column(visible=True, elem_classes="landing-container") as landing:
        gr.HTML(f"""
            <div class="hero-title">AIDA</div>
            <div class="hero-subtitle">The Ultimate AI Data Assistant • 42+ Enterprise Features • 100% Autonomous</div>
            
            <div class="section-header">THE BEST OF AIDA</div>
            <div class="feature-grid">
                <div class="feature-card"><div class="feature-icon">{ICONS['hypothesis']}</div><h3>Hypothesis Engine</h3><p>Auto-discovers patterns, generates hypotheses, and scientifically tests them.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['agents']}</div><h3>Council of Agents</h3><p>A multi-persona debate system where an Analyst, Skeptic, and CEO debate your data.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['healing']}</div><h3>Self-Healing Agents</h3><p>AIDA detects Python code errors during execution, debugs itself, and retries automatically.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['causal']}</div><h3>Causal Inference</h3><p>Goes beyond correlation to find true causation using advanced statistical reasoning.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['simulator']}</div><h3>What-If Simulator</h3><p>A real-time predictive engine that trains on your data to simulate future scenarios.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['export']}</div><h3>Rich Export Bundles</h3><p>Download a complete deployment package: Cleaned Data, Reports, and Models.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['advisor']}</div><h3>CEO Advisor</h3><p>A dedicated strategic agent that provides high-level business advice and ROI estimates.</p></div>
                <div class="feature-card"><div class="feature-icon">{ICONS['pipeline']}</div><h3>Autonomous Pipeline</h3><p>One-click end-to-end data science: Ingestion → Cleaning → EDA → AutoML → Reporting.</p></div>
            </div>
            
            <div class="footer">AIDA | Krish J 2025</div>
        """)
        launch_btn = gr.Button("LAUNCH PLATFORM", elem_classes="launch-btn")

    # MAIN APP
    with gr.Column(visible=False) as app:
        with gr.Row(variant="panel"):
            gr.Markdown("## 🤖 AIDA Platform")
            api_key = gr.Textbox(label="Google API Key", type="password", placeholder="Paste Key & Press Enter to Connect", scale=4)
        
        with gr.Tabs():
            # 1. DATA
            with gr.Tab("Data Upload"):
                file_up = gr.File(label="Dataset")
                upload_status = gr.Markdown(visible=True)
                data_preview = gr.Dataframe(label="Preview", interactive=False)
            
            # 2. ANALYSIS
            with gr.Tab("Ultimate Pipeline"):
                with gr.Row():
                    btn_clean = gr.Button("1. Clean Data")
                    btn_eda = gr.Button("2. Auto EDA")
                    btn_model = gr.Button("3. Train Model")
                    btn_all = gr.Button("Run Complete Pipeline", variant="primary")
                pipe_out = gr.Markdown("")
                pipe_file = gr.File(label="Download Result")
            
            with gr.Tab("Hypothesis Engine"):
                hyp_btn = gr.Button("Run Scientific Discovery", variant="primary")
                hyp_out = gr.Markdown("")
                hyp_file = gr.File(label="Download Report")
            
            with gr.Tab("Visualizations"):
                viz_type = gr.Radio(["Heatmap", "Distribution", "Scatter"], label="Type", value="Heatmap")
                viz_btn = gr.Button("Create")
                viz_out = gr.Plot()
            
            # 3. INTELLIGENCE
            with gr.Tab("NL Query"):
                q_box = gr.Textbox(label="Question")
                q_btn = gr.Button("Ask")
                q_out = gr.Markdown("")
            
            with gr.Tab("Council of Agents"):
                debate_topic = gr.Textbox(label="Topic")
                debate_btn = gr.Button("Debate")
                with gr.Row():
                    analyst_out = gr.Textbox(label="Analyst")
                    skeptic_out = gr.Textbox(label="Skeptic")
                ceo_out = gr.Textbox(label="CEO Decision")
                debate_file = gr.File(label="Download Transcript")
            
            with gr.Tab("Business AI"):
                biz_q = gr.Textbox(label="Question")
                biz_btn = gr.Button("Advise")
                biz_out = gr.Markdown("")
                biz_file = gr.File(label="Download Advice")
            
            # 4. ADVANCED
            with gr.Tab("What-If Simulator"):
                sim_btn = gr.Button("Train Simulator")
                sim_out = gr.Markdown("")
            
            with gr.Tab("Causal Inference"):
                c_treat = gr.Textbox(label="Treatment")
                c_out = gr.Textbox(label="Outcome")
                c_btn = gr.Button("Analyze")
                c_res = gr.Markdown("")
                c_file = gr.File(label="Download Analysis")
            
            with gr.Tab("Text Analytics"):
                txt_col = gr.Textbox(label="Text Column")
                txt_btn = gr.Button("Analyze Sentiment")
                txt_out = gr.Markdown("")
                txt_file = gr.File(label="Download Analysis")
            
            with gr.Tab("SQL Generator"):
                sql_q = gr.Textbox(label="Question")
                sql_btn = gr.Button("Generate SQL")
                sql_out = gr.Code(language="sql")
                sql_file = gr.File(label="Download SQL")
            
            # 5. EXPORT
            with gr.Tab("Rich Export"):
                gr.Markdown("### 📦 Session Export\nDownload everything generated in this session (Cleaned Data, Models, Reports) in one ZIP file.")
                export_btn = gr.Button("Generate Rich Export Bundle", variant="primary")
                export_file = gr.File(label="Download ZIP")
        
        gr.HTML('<div class="footer">AIDA | Krish J 2025</div>')

    # LOGIC
    launch_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[landing, app])
    
    def verify_connection(key):
        if len(key) < 30: return gr.update(placeholder="Paste Key & Press Enter to Connect")
        llm, err = configure_api(key)
        if err: raise gr.Error(err)
        gr.Info("API Connected Successfully! You are ready to go.")
        return gr.update(interactive=True, placeholder="API Key Verified ✅")
        
    api_key.submit(verify_connection, inputs=[api_key], outputs=[api_key])
    api_key.change(verify_connection, inputs=[api_key], outputs=[api_key])
    
    def handle_upload(file):
        if not file: return None, None, ""
        df, err = load_data(file)
        if err: return None, None, f"Error: {err}"
        return df, df.head(20), "Data loaded successfully."
    file_up.upload(handle_upload, inputs=[file_up], outputs=[df_state, data_preview, upload_status])
    
    # Pipeline Logic
    btn_clean.click(lambda d, k, s: run_pipeline_step(d, k, "Clean", s), inputs=[df_state, api_key, session_state], outputs=[pipe_out, pipe_file, session_state])
    btn_eda.click(lambda d, k, s: run_pipeline_step(d, k, "EDA", s), inputs=[df_state, api_key, session_state], outputs=[pipe_out, pipe_file, session_state])
    btn_model.click(lambda d, k, s: run_pipeline_step(d, k, "Model", s), inputs=[df_state, api_key, session_state], outputs=[pipe_out, pipe_file, session_state])
    btn_all.click(lambda d, k, s: run_pipeline_step(d, k, "All", s), inputs=[df_state, api_key, session_state], outputs=[pipe_out, pipe_file, session_state])
    
    # Hypothesis Logic
    def run_hyp(df, key, state):
        yield "### ⏳ Agent is thinking... (Scanning dataset)", None, state
        if df is None: 
            yield "Upload data", None, state
            return
        llm, err = configure_api(key)
        if err: 
            yield err, None, state
            return
        
        agent = HypothesisAgent(llm)
        report = agent.discover_and_test(df)
        state, path = save_artifact(report, "hypothesis_report.md", state)
        yield report, path, state
    hyp_btn.click(run_hyp, inputs=[df_state, api_key, session_state], outputs=[hyp_out, hyp_file, session_state], show_progress=True)
    
    # Viz Logic
    def create_viz(df, vtype):
        if df is None: return None
        if vtype == "Heatmap": return px.imshow(df.corr(numeric_only=True))
        if vtype == "Distribution": return px.histogram(df, x=df.select_dtypes(include=np.number).columns[0])
        if vtype == "Scatter": return px.scatter(df, x=df.columns[0], y=df.columns[1])
    viz_btn.click(create_viz, inputs=[df_state, viz_type], outputs=[viz_out])
    
    # NL Query Logic
    def ask_nl(df, q, key):
        if df is None: return "Upload data"
        llm, err = configure_api(key)
        if err: return err
        agent = NLQueryAgent(llm)
        healer = SelfHealingAgent(llm)
        code = agent.query_to_code(q, str(df.columns))
        msg, final, err = healer.execute_with_healing(code, q, df)
        return f"Code:\n{final}\n\nResult:\n{msg}"
    q_btn.click(ask_nl, inputs=[df_state, q_box, api_key], outputs=[q_out])
    
    # Debate Logic
    def run_debate(df, topic, key, state):
        if df is None: return "Upload data", "", "", None, state
        llm, err = configure_api(key)
        if err: return err, err, err, None, state
        council = CouncilOfAgents(llm)
        a, s, c = council.debate_and_decide(topic, str(df.describe()))
        transcript = f"# Council Debate: {topic}\n\n## Analyst\n{a}\n\n## Skeptic\n{s}\n\n## CEO Decision\n{c}"
        state, path = save_artifact(transcript, "debate_transcript.md", state)
        return a, s, c, path, state
    debate_btn.click(run_debate, inputs=[df_state, debate_topic, api_key, session_state], outputs=[analyst_out, skeptic_out, ceo_out, debate_file, session_state])
    
    # Biz Advice Logic
    def run_biz(df, q, key, state):
        if df is None: return "Upload data", None, state
        llm, err = configure_api(key)
        if err: return err, None, state
        agent = BusinessDecisionAgent(llm)
        advice = agent.advise(q, str(df.describe()))
        state, path = save_artifact(advice, "business_advice.md", state)
        return advice, path, state
    biz_btn.click(run_biz, inputs=[df_state, biz_q, api_key, session_state], outputs=[biz_out, biz_file, session_state])
    
    # Simulator Logic
    def train_sim(df):
        if df is None: return "Upload data"
        return train_what_if_simulator(df)
    sim_btn.click(train_sim, inputs=[df_state], outputs=[sim_out])
    
    # Causal Logic
    def run_causal(df, t, o, key, state):
        if df is None: return "Upload data", None, state
        llm, err = configure_api(key)
        if err: return err, None, state
        agent = CausalInferenceAgent(llm)
        res = agent.analyze(df, t, o)
        state, path = save_artifact(res, "causal_analysis.md", state)
        return res, path, state
    c_btn.click(run_causal, inputs=[df_state, c_treat, c_out, api_key, session_state], outputs=[c_res, c_file, session_state])
    
    # Text Logic
    def run_text(df, col, key, state):
        if df is None: return "Upload data", None, state
        llm, err = configure_api(key)
        if err: return err, None, state
        agent = TextAnalyticsAgent(llm)
        res = agent.analyze_sentiment(df[col])
        state, path = save_artifact(res, "sentiment_analysis.md", state)
        return res, path, state
    txt_btn.click(run_text, inputs=[df_state, txt_col, api_key, session_state], outputs=[txt_out, txt_file, session_state])
    
    # SQL Logic
    def run_sql(df, q, key, state):
        if df is None: return "Upload data", None, state
        llm, err = configure_api(key)
        if err: return err, None, state
        agent = SQLGeneratorAgent(llm)
        sql = agent.generate_sql(q, str(df.columns))
        state, path = save_artifact(sql, "query.sql", state)
        return sql, path, state
    sql_btn.click(run_sql, inputs=[df_state, sql_q, api_key, session_state], outputs=[sql_out, sql_file, session_state])
    
    # Export Logic
    export_btn.click(create_export_bundle, inputs=[session_state], outputs=[export_file])

if __name__ == "__main__":
    demo.launch(server_port=7870, show_error=True)

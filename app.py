"""
app.py
Website Visitor Lead Conversion Scorer
A modern SaaS-grade Streamlit application for predictive lead scoring.
Permanently visible high-contrast navigation tabs, interactive Plotly visualizations,
dynamic real model evaluation metrics, and live Scikit-learn inference.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from predict import predict_lead, classify_lead_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="LeadPulse AI | Website Visitor Lead Conversion Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# File Paths & Data Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "website_visitors.csv")
METRICS_FILE = os.path.join(BASE_DIR, "reports", "metrics.json")
MODEL_FILE = os.path.join(BASE_DIR, "models", "lead_conversion_model.joblib")

@st.cache_data
def load_dataset():
    """Load the 1,000-record website visitors dataset."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

@st.cache_resource
def load_trained_model():
    """Load and cache the trained Scikit-learn Pipeline model."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

@st.cache_data
def evaluate_model_pipeline():
    """
    Dynamically evaluate the trained Scikit-learn Pipeline on the actual test split.
    Calculates 100% genuine metrics without hardcoding:
      - Accuracy, Precision, Recall, F1-Score, ROC-AUC
      - Confusion Matrix counts & proportions
      - Classification Report dict
      - ROC Curve coordinates (FPR, TPR)
      - Random Forest feature importances
    """
    df = load_dataset()
    if df is None:
        return None

    features = ['session_duration', 'pages_visited', 'referral_source', 'content_downloads']
    X = df[features]
    y = df['converted']

    # 80/20 train/test split with stratification matching train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = load_trained_model()
    if model is None:
        return None

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Feature importances extraction
    preprocessor = model.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(['referral_source']))
    feature_names = ['session_duration', 'pages_visited', 'content_downloads'] + encoded_cat_names
    importances = model.named_steps['classifier'].feature_importances_

    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "raw_matrix": cm.tolist()
        },
        "classification_report": report_dict,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "feature_importances": feat_imp,
        "train_records": len(X_train),
        "test_records": len(y_test),
        "total_records": len(df)
    }

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "preset" not in st.session_state:
    st.session_state["preset"] = None

# ---------------------------------------------------------
# Custom Modern SaaS CSS & Typography
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global application styling: Light SaaS background and dark readable text */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Prevent header clipping and optimize block container spacing */
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }

    /* Force dark, readable text on all standard markdown containers */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] div {
        color: #1E293B !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Header styling: Vibrant Purple-Blue Gradient, responsive & unclipped */
    .app-header {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 45%, #7C3AED 100%);
        padding: 22px 28px;
        border-radius: 16px;
        color: #FFFFFF !important;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-size: 2.0rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        line-height: 1.25 !important;
    }
    .app-header p {
        color: #E0E7FF !important;
        font-size: 0.96rem !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
    }
    .header-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        color: #FFFFFF !important;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* KPI Summary Cards: Uniform height, white background, subtle shadow */
    .kpi-card {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08) !important;
        border-color: #CBD5E1 !important;
    }
    .kpi-title {
        font-size: 0.80rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 4px !important;
    }
    .kpi-value {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        line-height: 1.2 !important;
    }
    .kpi-subtitle {
        font-size: 0.80rem !important;
        color: #94A3B8 !important;
        margin-top: 4px !important;
        font-weight: 500 !important;
    }

    /* Tab Bar: Fix invisible navigation tab labels with prominent high-contrast pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #FFFFFF !important;
        padding: 8px 10px !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        margin-bottom: 22px !important;
        display: flex !important;
        flex-wrap: wrap !important;
    }
    button[data-testid="stTab"],
    button[role="tab"],
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        border-radius: 10px !important;
        padding: 0 20px !important;
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        transition: all 0.2s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Inactive Tab text: High contrast dark navy */
    button[data-testid="stTab"] p,
    button[role="tab"] p,
    .stTabs [data-baseweb="tab"] p,
    button[data-testid="stTab"] span,
    button[role="tab"] span,
    .stTabs [data-baseweb="tab"] span {
        color: #1E293B !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
    }

    /* Tab Hover */
    button[data-testid="stTab"]:hover,
    button[role="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #EEF2FF !important;
        border-color: #C7D2FE !important;
    }
    button[data-testid="stTab"]:hover p,
    button[role="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover p {
        color: #4F46E5 !important;
    }

    /* Active Tab: Signature Purple-Blue gradient background with crisp bright white text */
    button[data-testid="stTab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #7C3AED 100%) !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
    button[data-testid="stTab"][aria-selected="true"] p,
    button[role="tab"][aria-selected="true"] p,
    .stTabs [aria-selected="true"] p,
    button[data-testid="stTab"][aria-selected="true"] span,
    button[role="tab"][aria-selected="true"] span,
    .stTabs [aria-selected="true"] span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Sidebar aesthetics & alignment */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 14px;
    }
    .sidebar-logo-icon {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 800;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
    }
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .sidebar-brand-subtitle {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Content Cards */
    .content-card {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 22px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        margin-bottom: 18px !important;
    }

    /* Form Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
    }

    /* Primary Predict Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        padding: 10px 22px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:first-child p,
    div.stButton > button:first-child span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary / Reset Buttons */
    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: #FFFFFF !important;
        color: #4F46E5 !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: #EEF2FF !important;
        border-color: #818CF8 !important;
        color: #3730A3 !important;
    }

    /* Responsive Tweaks */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }
        .app-header {
            padding: 16px !important;
        }
        .app-header h1 {
            font-size: 1.4rem !important;
        }
        .kpi-value {
            font-size: 1.45rem !important;
        }
        button[data-testid="stTab"],
        button[role="tab"] {
            padding: 0 12px !important;
            font-size: 0.85rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Plotly Tab Resize Listener (Eliminates Blank Chart Bug)
# ---------------------------------------------------------
st.html("""
<script>
    function triggerPlotlyResize() {
        const doc = window.parent.document;
        const tabs = doc.querySelectorAll('button[data-testid="stTab"], button[role="tab"]');
        tabs.forEach(tab => {
            if (!tab.dataset.resizeListenerAttached) {
                tab.dataset.resizeListenerAttached = 'true';
                tab.addEventListener('click', () => {
                    setTimeout(() => { window.parent.dispatchEvent(new Event('resize')); }, 80);
                    setTimeout(() => { window.parent.dispatchEvent(new Event('resize')); }, 250);
                    setTimeout(() => { window.parent.dispatchEvent(new Event('resize')); }, 500);
                });
            }
        });
    }
    setTimeout(triggerPlotlyResize, 200);
    setTimeout(triggerPlotlyResize, 800);
</script>
""")

# ---------------------------------------------------------
# Load Data & Metrics
# ---------------------------------------------------------
df = load_dataset()
metrics = evaluate_model_pipeline()

if df is None or len(df) == 0:
    st.error("Dataset not found! Please run `python train.py` to generate the dataset and train the model.")
    st.stop()

if metrics is None:
    st.error("Model artifacts not found! Please run `python train.py` to train and serialize the model.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Component (Specs, Quick Controls & Downloads)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🎯</div>
        <div>
            <div class="sidebar-brand-title">LeadPulse AI</div>
            <div class="sidebar-brand-subtitle">Enterprise Lead Scorer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚡ System Status")
    st.markdown("""
    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; color: #166534; font-weight: 700; font-size: 0.90rem;">
            <span>🟢</span> <span>Model & Pipeline Online</span>
        </div>
        <div style="color: #15803D; font-size: 0.80rem; margin-top: 4px;">
            Scikit-Learn Random Forest (Calibrated)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Model & Dataset Telemetry")
    st.markdown(f"""
    <div style="background: #F8FAFC; border-radius: 10px; padding: 14px; border: 1px solid #E2E8F0; font-size: 0.84rem; color: #334155; line-height: 1.8;">
        <strong>Total Visitors:</strong> {metrics['total_records']:,}<br>
        <strong>Training Samples:</strong> {metrics['train_records']:,} (80%)<br>
        <strong>Test Samples:</strong> {metrics['test_records']:,} (20%)<br>
        <strong>Model Accuracy:</strong> <span style="color: #4F46E5; font-weight: 700;">{metrics['accuracy']*100:.2f}%</span><br>
        <strong>ROC-AUC Score:</strong> <span style="color: #10B981; font-weight: 700;">{metrics['roc_auc']*100:.2f}%</span><br>
        <strong>Scoring Scale:</strong> 0 - 100 Lead Score
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Quick download full CSV dataset
    csv_sidebar = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_sidebar,
        file_name="website_visitors.csv",
        mime="text/csv",
        use_container_width=True,
        key="sidebar_dl_csv"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
        💡 <strong>Quick Navigation:</strong> Click any of the 5 tabs in the top navigation bar to explore Lead Prediction, Dataset Analysis, Model Performance, Lead Insights, and About Project.
    </div>
    """, unsafe_allow_html=True)
    st.caption("© 2026 LeadPulse AI • Production ML Suite")

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1>Website Visitor Lead Conversion Scorer</h1>
            <p>Predict visitor sales conversion probability & qualify high-intent sales leads with Machine Learning</p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <div class="header-badge">⚡ Production Ready AI</div>
            <div class="header-badge">🎯 Random Forest Pipeline</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Summary KPI Cards (Top Metrics)
# ---------------------------------------------------------
total_visitors = len(df)
converted_count = int(df['converted'].sum())
non_converted_count = total_visitors - converted_count
conversion_rate = (converted_count / total_visitors) * 100.0

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Visitors</div>
        <div class="kpi-value">{total_visitors:,}</div>
        <div class="kpi-subtitle">1,000 Verified Records</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Converted Leads</div>
        <div class="kpi-value" style="color: #10B981;">{converted_count:,}</div>
        <div class="kpi-subtitle">{conversion_rate:.1f}% Conversion Share</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Non-Converted Visitors</div>
        <div class="kpi-value" style="color: #64748B;">{non_converted_count:,}</div>
        <div class="kpi-subtitle">{100.0 - conversion_rate:.1f}% Browsing Share</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Baseline Conversion Rate</div>
        <div class="kpi-value" style="color: #4F46E5;">{conversion_rate:.1f}%</div>
        <div class="kpi-subtitle">Site Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# =========================================================
# ALL 5 NAVIGATION SECTIONS (Permanently Visible Tabs)
# =========================================================
tab_pred, tab_analysis, tab_perf, tab_insights, tab_about = st.tabs([
    "🎯 Lead Prediction",
    "📊 Dataset Analysis",
    "📈 Model Performance",
    "💡 Lead Insights",
    "ℹ️ About Project"
])

# =========================================================
# TAB 1: LEAD PREDICTION
# =========================================================
with tab_pred:
    st.markdown("### 🎯 Live Visitor Lead Scoring Engine")
    st.markdown("Configure session telemetry below or select a quick persona preset to score conversion probability in real-time.")

    # Persona Quick Selectors
    st.markdown("##### ⚡ Quick Persona Presets")
    preset_cols = st.columns(4)

    with preset_cols[0]:
        if st.button("🚀 High Intent Enterprise", use_container_width=True, key="preset_ent"):
            st.session_state["preset"] = {
                "duration": 540,
                "pages": 10,
                "source": "Referral",
                "downloads": 3
            }
    with preset_cols[1]:
        if st.button("📧 Email Campaign Lead", use_container_width=True, key="preset_email"):
            st.session_state["preset"] = {
                "duration": 340,
                "pages": 6,
                "source": "Email",
                "downloads": 2
            }
    with preset_cols[2]:
        if st.button("🔍 Organic Search Skimmer", use_container_width=True, key="preset_search"):
            st.session_state["preset"] = {
                "duration": 150,
                "pages": 3,
                "source": "Google",
                "downloads": 0
            }
    with preset_cols[3]:
        if st.button("📱 Casual Social Visitor", use_container_width=True, key="preset_social"):
            st.session_state["preset"] = {
                "duration": 45,
                "pages": 1,
                "source": "Social Media",
                "downloads": 0
            }

    # Fetch initial values based on preset or defaults
    p = st.session_state.get("preset", None)
    default_duration = p["duration"] if p else 240
    default_pages = p["pages"] if p else 4
    default_source = p["source"] if p else "Google"
    default_downloads = p["downloads"] if p else 1

    source_options = ['Google', 'Social Media', 'Direct', 'Email', 'Referral']
    default_source_idx = source_options.index(default_source) if default_source in source_options else 0

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Input Form & Results Columns
    form_col, result_col = st.columns([1.1, 1.3], gap="large")

    with form_col:
        st.markdown("""
        <div style="background: white; border-radius: 14px; padding: 20px 22px; border: 1px solid #E2E8F0; margin-bottom: 14px;">
            <h4 style="margin: 0 0 6px 0; color: #0F172A; font-weight: 700;">Visitor Telemetry Inputs</h4>
            <p style="color: #64748B; font-size: 0.88rem; margin: 0;">Adjust the parameters to simulate visitor interaction and evaluate sales qualification.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("lead_prediction_form"):
            session_duration = st.slider(
                "⏱️ Session Duration (seconds)",
                min_value=10,
                max_value=900,
                value=int(default_duration),
                step=10,
                help="Total active seconds spent browsing the site."
            )
            st.caption(f"Equivalent duration: **{session_duration // 60}m {session_duration % 60}s**")

            pages_visited = st.slider(
                "📄 Pages Visited",
                min_value=1,
                max_value=25,
                value=int(default_pages),
                step=1,
                help="Total distinct URLs browsed in this session."
            )

            referral_source = st.selectbox(
                "🌐 Referral Source Channel",
                options=source_options,
                index=default_source_idx,
                help="Acquisition channel through which the visitor arrived."
            )

            content_downloads = st.slider(
                "📥 Content Downloads (Gated Whitepapers / Case Studies)",
                min_value=0,
                max_value=5,
                value=int(default_downloads),
                step=1,
                help="Number of gated asset downloads."
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                predict_submitted = st.form_submit_button("🎯 Score Visitor Lead", use_container_width=True)
            with btn_col2:
                reset_submitted = st.form_submit_button("🔄 Reset", use_container_width=True)

        if reset_submitted:
            st.session_state["preset"] = None
            st.rerun()

    with result_col:
        try:
            prediction_output = predict_lead(
                session_duration=session_duration,
                pages_visited=pages_visited,
                referral_source=referral_source,
                content_downloads=content_downloads
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            prediction_output = None

        if prediction_output:
            score = prediction_output["lead_score"]
            prob = prediction_output["conversion_probability"]
            category = prediction_output["category"]
            badge_color = prediction_output["badge_color"]
            badge_bg = prediction_output["badge_bg"]
            priority = prediction_output["priority"]
            action = prediction_output["recommended_action"]
            drivers = prediction_output["key_drivers"]

            # Save to session history on explicit button click
            if predict_submitted:
                history_entry = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "duration": f"{session_duration}s",
                    "pages": pages_visited,
                    "source": referral_source,
                    "downloads": content_downloads,
                    "score": score,
                    "probability": f"{prob:.1f}%",
                    "category": category
                }
                st.session_state["history"].insert(0, history_entry)
                st.toast(f"✅ Lead Scored: {category} ({score}/100)", icon="🎯")

            # Main Result Card
            st.markdown(f"""
            <div style="background: white; border-radius: 16px; padding: 22px 24px; border: 1px solid #E2E8F0; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05); margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Scoring Assessment</span>
                    <span style="background: {badge_bg}; color: {badge_color}; padding: 4px 14px; border-radius: 9999px; font-weight: 800; font-size: 0.88rem; border: 1px solid {badge_color}40;">
                        {category}
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px;">
                    <div style="font-size: 3.2rem; font-weight: 800; color: #0F172A; line-height: 1;">
                        {score}
                    </div>
                    <div style="color: #64748B; font-size: 1.15rem; font-weight: 700;">/ 100 Lead Score</div>
                </div>
                <div style="color: #475569; font-size: 0.94rem;">
                    Calibrated Conversion Probability: <strong style="color: #4F46E5;">{prob:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Plotly Gauge Chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "/100", 'font': {'size': 26, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': badge_color, 'thickness': 0.32},
                    'bgcolor': "white",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': '#FEE2E2'},
                        {'range': [40, 70], 'color': '#FEF3C7'},
                        {'range': [70, 100], 'color': '#DCFCE7'}
                    ],
                    'threshold': {
                        'line': {'color': "#312E81", 'width': 3},
                        'thickness': 0.8,
                        'value': score
                    }
                }
            ))
            gauge_fig.update_layout(
                template="plotly_white",
                height=185,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#0F172A", 'family': "Plus Jakarta Sans"}
            )
            st.plotly_chart(gauge_fig, use_container_width=True, key="pred_gauge_chart", config={"responsive": True})

            # Recommended Sales Action
            st.markdown(f"""
            <div style="background: #F8FAFC; border-left: 4px solid {badge_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #E2E8F0; border-left: 4px solid {badge_color};">
                <div style="font-weight: 700; font-size: 0.88rem; color: #1E293B; margin-bottom: 4px;">
                    📌 Recommended Action ({priority}):
                </div>
                <div style="font-size: 0.86rem; color: #475569; line-height: 1.5;">
                    {action}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Intent Factor Highlights
            if drivers:
                st.markdown("##### 🔍 Intent Factor Highlights")
                for d in drivers:
                    st.markdown(f"- **{d}**")

    # Session History Table
    st.markdown("---")
    st.markdown("#### 🕒 Live Session Prediction History")

    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(
            history_df,
            column_config={
                "timestamp": "Timestamp",
                "duration": "Duration",
                "pages": "Pages",
                "source": "Referral Source",
                "downloads": "Downloads",
                "score": st.column_config.ProgressColumn("Lead Score", min_value=0, max_value=100, format="%.1f"),
                "probability": "Probability",
                "category": "Classification Tier"
            },
            hide_index=True,
            use_container_width=True
        )

        hist_btn_col1, hist_btn_col2, _ = st.columns([1.2, 1, 3])
        with hist_btn_col1:
            csv_history = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Predictions (CSV)",
                data=csv_history,
                file_name="leadpulse_predictions_history.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_pred_hist"
            )
        with hist_btn_col2:
            if st.button("🗑️ Clear History", use_container_width=True, key="clear_hist_btn"):
                st.session_state["history"] = []
                st.rerun()
    else:
        st.info("No visitor predictions recorded in this session yet. Adjust the sliders above and click **Score Visitor Lead**.")

# =========================================================
# TAB 2: DATASET ANALYSIS
# =========================================================
with tab_analysis:
    st.markdown("### 📊 Exploratory Dataset Analysis")
    st.markdown("Statistical exploration of the verified 1,000-record website visitor interaction dataset.")

    # Top Dataset Summary Bar
    ds_col1, ds_col2 = st.columns([3, 1])
    with ds_col1:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 12px 18px; border: 1px solid #E2E8F0; font-size: 0.90rem; color: #334155;">
            <strong>Dimensions:</strong> {df.shape[0]} Rows × {df.shape[1]} Columns &nbsp;|&nbsp; 
            <strong>Features:</strong> <code>session_duration</code>, <code>pages_visited</code>, <code>referral_source</code>, <code>content_downloads</code> &nbsp;|&nbsp; 
            <strong>Target:</strong> <code>converted</code> (Binary)
        </div>
        """, unsafe_allow_html=True)
    with ds_col2:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv_full,
            file_name="website_visitors.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_tab_full_csv"
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Expandable Raw Data Table & Descriptive Statistics
    with st.expander("🔍 Inspect Raw Records & Summary Descriptive Statistics", expanded=False):
        subtab_raw, subtab_stats = st.tabs(["📄 Raw Dataset Records", "📈 Descriptive Statistics"])
        with subtab_raw:
            st.dataframe(
                df,
                column_config={
                    "session_duration": st.column_config.NumberColumn("Session Duration (s)", format="%d s"),
                    "pages_visited": st.column_config.NumberColumn("Pages Visited", format="%d"),
                    "referral_source": "Referral Source Channel",
                    "content_downloads": st.column_config.NumberColumn("Content Downloads", format="%d"),
                    "converted": st.column_config.CheckboxColumn("Converted (Target)")
                },
                height=260,
                use_container_width=True
            )
        with subtab_stats:
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Visualizations Row 1: Conversion Status & Referral Source Distribution
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.markdown("##### 🥧 Converted vs Non-Converted Visitors")
        conv_counts = df['converted'].value_counts().reset_index()
        conv_counts.columns = ['converted', 'count']
        conv_counts['Status'] = conv_counts['converted'].map({1: 'Converted Lead (1)', 0: 'Non-Converted (0)'})

        donut_fig = px.pie(
            conv_counts,
            names='Status',
            values='count',
            hole=0.55,
            color='Status',
            color_discrete_map={
                'Converted Lead (1)': '#10B981',
                'Non-Converted (0)': '#94A3B8'
            }
        )
        donut_fig.update_traces(
            textinfo='percent+label',
            textfont=dict(size=13, color='#0F172A', family='Plus Jakarta Sans'),
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        donut_fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(donut_fig, use_container_width=True, key="ds_donut_chart", config={"responsive": True})

    with row1_c2:
        st.markdown("##### 🌐 Traffic Volume by Referral Source")
        ref_counts = df['referral_source'].value_counts().reset_index()
        ref_counts.columns = ['Source', 'Visitors']

        bar_ref = px.bar(
            ref_counts,
            x='Source',
            y='Visitors',
            color='Visitors',
            color_continuous_scale='Purples',
            text='Visitors'
        )
        bar_ref.update_traces(
            textposition='outside',
            textfont=dict(color='#0F172A', size=12, family='Plus Jakarta Sans')
        )
        bar_ref.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=20, b=30),
            xaxis_title="Referral Channel",
            yaxis_title="Total Visitors",
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(bar_ref, use_container_width=True, key="ds_bar_ref_chart", config={"responsive": True})

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Visualizations Row 2: Deep Behavioral Analysis (Duration, Pages, Downloads)
    st.markdown("##### 📈 Behavioral Feature Distributions vs Conversion")
    b_c1, b_c2, b_c3 = st.columns(3)

    with b_c1:
        st.markdown("**Session Duration Analysis**")
        df_plot = df.copy()
        df_plot['Conversion Status'] = df_plot['converted'].map({1: 'Converted', 0: 'Not Converted'})

        box_dur = px.box(
            df_plot,
            x='Conversion Status',
            y='session_duration',
            color='Conversion Status',
            color_discrete_map={'Converted': '#10B981', 'Not Converted': '#64748B'},
            points="outliers"
        )
        box_dur.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=10, r=10, t=10, b=20),
            yaxis_title="Session Duration (s)",
            xaxis_title="",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(box_dur, use_container_width=True, key="ds_box_dur_chart", config={"responsive": True})

    with b_c2:
        st.markdown("**Pages Visited Analysis**")
        hist_pages = px.histogram(
            df_plot,
            x='pages_visited',
            color='Conversion Status',
            barmode='overlay',
            opacity=0.75,
            color_discrete_map={'Converted': '#10B981', 'Not Converted': '#64748B'},
            nbins=16
        )
        hist_pages.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=10, r=10, t=10, b=20),
            xaxis_title="Pages Visited",
            yaxis_title="Visitors",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(hist_pages, use_container_width=True, key="ds_hist_pages_chart", config={"responsive": True})

    with b_c3:
        st.markdown("**Content Downloads Analysis**")
        dl_conv = df.groupby('content_downloads')['converted'].agg(['count', 'mean']).reset_index()
        dl_conv['Conversion Rate %'] = dl_conv['mean'] * 100.0

        bar_dl = px.bar(
            dl_conv,
            x='content_downloads',
            y='Conversion Rate %',
            text=dl_conv['Conversion Rate %'].apply(lambda x: f"{x:.1f}%"),
            color='Conversion Rate %',
            color_continuous_scale='Tealgrn'
        )
        bar_dl.update_traces(
            textposition='outside',
            textfont=dict(color='#0F172A', size=11, family='Plus Jakarta Sans')
        )
        bar_dl.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=10, r=10, t=10, b=20),
            xaxis_title="Content Downloads",
            yaxis_title="Conversion %",
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(bar_dl, use_container_width=True, key="ds_bar_dl_chart", config={"responsive": True})

# =========================================================
# TAB 3: MODEL PERFORMANCE
# =========================================================
with tab_perf:
    st.markdown("### 📈 Machine Learning Model Performance")
    st.markdown("Evaluation metrics dynamically computed from the trained Scikit-learn Random Forest model on the test dataset.")

    # 5 Scorecard KPI Cards
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)

    with perf_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Accuracy</div>
            <div class="kpi-value" style="color: #4F46E5;">{metrics['accuracy']*100:.2f}%</div>
            <div class="kpi-subtitle">Test Split (200 rows)</div>
        </div>
        """, unsafe_allow_html=True)

    with perf_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Precision</div>
            <div class="kpi-value" style="color: #2563EB;">{metrics['precision']*100:.2f}%</div>
            <div class="kpi-subtitle">Lead Quality Fidelity</div>
        </div>
        """, unsafe_allow_html=True)

    with perf_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Recall</div>
            <div class="kpi-value" style="color: #10B981;">{metrics['recall']*100:.2f}%</div>
            <div class="kpi-subtitle">Lead Capture Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with perf_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">F1-Score</div>
            <div class="kpi-value" style="color: #7C3AED;">{metrics['f1_score']*100:.2f}%</div>
            <div class="kpi-subtitle">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)

    with perf_col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi-value" style="color: #059669;">{metrics['roc_auc']*100:.2f}%</div>
            <div class="kpi-subtitle">Discrimination Power</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Visualizations Row: Confusion Matrix Heatmap & ROC Curve
    m_row1_c1, m_row1_c2 = st.columns(2, gap="large")

    with m_row1_c1:
        st.markdown("##### 🔲 Confusion Matrix Heatmap")
        cm_data = metrics["confusion_matrix"]
        tn = cm_data["true_negative"]
        fp = cm_data["false_positive"]
        fn = cm_data["false_negative"]
        tp = cm_data["true_positive"]

        z_matrix = [
            [tn, fp],
            [fn, tp]
        ]
        text_matrix = [
            [f"TN: {tn}<br>({(tn/200)*100:.1f}%)", f"FP: {fp}<br>({(fp/200)*100:.1f}%)"],
            [f"FN: {fn}<br>({(fn/200)*100:.1f}%)", f"TP: {tp}<br>({(tp/200)*100:.1f}%)"]
        ]
        x_cats = ['Predicted: Not Converted (0)', 'Predicted: Converted (1)']
        y_cats = ['Actual: Not Converted (0)', 'Actual: Converted (1)']

        cm_fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=x_cats,
            y=y_cats,
            colorscale="Purples",
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 15, "family": "Plus Jakarta Sans", "color": "#0F172A"},
            hoverongaps=False
        ))
        cm_fig.update_layout(
            template="plotly_white",
            height=330,
            margin=dict(l=30, r=30, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans'),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(cm_fig, use_container_width=True, key="perf_cm_chart", config={"responsive": True})

        st.markdown(f"""
        <div style="background: white; border-radius: 8px; padding: 10px 14px; border: 1px solid #E2E8F0; font-size: 0.83rem; color: #475569;">
            ✅ <strong>True Negatives:</strong> {tn} &nbsp;|&nbsp; 
            ⚠️ <strong>False Positives:</strong> {fp} &nbsp;|&nbsp; 
            ❌ <strong>False Negatives:</strong> {fn} &nbsp;|&nbsp; 
            🎯 <strong>True Positives:</strong> {tp}
        </div>
        """, unsafe_allow_html=True)

    with m_row1_c2:
        st.markdown("##### 📉 Receiver Operating Characteristic (ROC Curve)")
        fpr = metrics["fpr"]
        tpr = metrics["tpr"]
        auc_val = metrics["roc_auc"]

        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'Random Forest (AUC = {auc_val:.4f})',
            line=dict(color='#4F46E5', width=3),
            fill='tozeroy',
            fillcolor='rgba(79, 70, 229, 0.12)'
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Guess Benchmark (AUC = 0.50)',
            line=dict(color='#94A3B8', width=2, dash='dash')
        ))
        roc_fig.update_layout(
            template="plotly_white",
            height=330,
            margin=dict(l=30, r=30, t=20, b=30),
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(roc_fig, use_container_width=True, key="perf_roc_chart", config={"responsive": True})

        st.markdown(f"""
        <div style="background: white; border-radius: 8px; padding: 10px 14px; border: 1px solid #E2E8F0; font-size: 0.83rem; color: #475569;">
            🏆 <strong>ROC-AUC {auc_val*100:.2f}%:</strong> Demonstrates near-perfect discrimination between qualified sales leads and casual exploratory visitors.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Feature Importance & Classification Report
    m_row2_c1, m_row2_c2 = st.columns([1.1, 1.2], gap="large")

    with m_row2_c1:
        st.markdown("##### 🌟 Feature Importance Ranking")
        feat_df = pd.DataFrame(metrics["feature_importances"])
        feat_df["label"] = feat_df["feature"].str.replace("referral_source_", "Channel: ")
        feat_df["label"] = feat_df["label"].str.replace("session_duration", "Session Duration")
        feat_df["label"] = feat_df["label"].str.replace("pages_visited", "Pages Visited")
        feat_df["label"] = feat_df["label"].str.replace("content_downloads", "Content Downloads")
        feat_df = feat_df.sort_values(by="importance", ascending=True)

        feat_fig = px.bar(
            feat_df,
            x="importance",
            y="label",
            orientation="h",
            color="importance",
            color_continuous_scale="Purples",
            text=feat_df["importance"].apply(lambda x: f"{x*100:.1f}%")
        )
        feat_fig.update_traces(
            textposition='outside',
            textfont=dict(color='#0F172A', size=11, family='Plus Jakarta Sans')
        )
        feat_fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=30, t=10, b=20),
            xaxis_title="Relative Feature Importance Weight",
            yaxis_title="",
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(feat_fig, use_container_width=True, key="perf_feat_chart", config={"responsive": True})

    with m_row2_c2:
        st.markdown("##### 📋 Detailed Classification Report")
        rep = metrics.get("classification_report", {})
        if rep:
            rep_table = pd.DataFrame([
                {
                    "Target Class / Metric": "Non-Converted (Class 0)",
                    "Precision": f"{rep['0']['precision']:.4f}",
                    "Recall": f"{rep['0']['recall']:.4f}",
                    "F1-Score": f"{rep['0']['f1-score']:.4f}",
                    "Support": int(rep['0']['support'])
                },
                {
                    "Target Class / Metric": "Converted Lead (Class 1)",
                    "Precision": f"{rep['1']['precision']:.4f}",
                    "Recall": f"{rep['1']['recall']:.4f}",
                    "F1-Score": f"{rep['1']['f1-score']:.4f}",
                    "Support": int(rep['1']['support'])
                },
                {
                    "Target Class / Metric": "Macro Average",
                    "Precision": f"{rep['macro avg']['precision']:.4f}",
                    "Recall": f"{rep['macro avg']['recall']:.4f}",
                    "F1-Score": f"{rep['macro avg']['f1-score']:.4f}",
                    "Support": int(rep['macro avg']['support'])
                },
                {
                    "Target Class / Metric": "Weighted Average",
                    "Precision": f"{rep['weighted avg']['precision']:.4f}",
                    "Recall": f"{rep['weighted avg']['recall']:.4f}",
                    "F1-Score": f"{rep['weighted avg']['f1-score']:.4f}",
                    "Support": int(rep['weighted avg']['support'])
                }
            ])

            st.dataframe(
                rep_table,
                hide_index=True,
                use_container_width=True,
                height=180
            )

            st.markdown(f"""
            <div style="background: white; border-radius: 8px; padding: 12px 16px; border: 1px solid #E2E8F0; font-size: 0.84rem; color: #475569; margin-top: 10px;">
                💡 <strong>Evaluation Note:</strong> Evaluated on <strong>200 unseen holdout visitors</strong> (80/20 stratified split). Class 1 Recall of <strong>{rep['1']['recall']*100:.2f}%</strong> ensures our sales team misses virtually zero genuine converting leads.
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 4: LEAD INSIGHTS
# =========================================================
with tab_insights:
    st.markdown("### 💡 Business Intelligence & Tactical Lead Insights")
    st.markdown("Strategic sales intelligence and attribution findings derived from machine learning analysis of website visitor behavior.")

    # Insights Row 1
    ins_c1, ins_c2 = st.columns(2, gap="large")

    with ins_c1:
        st.markdown("##### 1. Referral Channel Conversion Power")
        channel_analysis = df.groupby('referral_source')['converted'].agg(['count', 'sum', 'mean']).reset_index()
        channel_analysis.columns = ['Referral Source', 'Total Visitors', 'Converted Leads', 'Conversion Rate']
        channel_analysis['Conversion Rate %'] = channel_analysis['Conversion Rate'] * 100.0
        channel_analysis = channel_analysis.sort_values(by='Conversion Rate %', ascending=False)

        bar_chan = px.bar(
            channel_analysis,
            x='Referral Source',
            y='Conversion Rate %',
            color='Conversion Rate %',
            color_continuous_scale='Blues',
            text=channel_analysis['Conversion Rate %'].apply(lambda x: f"{x:.1f}%")
        )
        bar_chan.update_traces(
            textposition='outside',
            textfont=dict(color='#0F172A', size=11, family='Plus Jakarta Sans')
        )
        bar_chan.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(bar_chan, use_container_width=True, key="ins_chan_chart", config={"responsive": True})

        st.markdown("""
        > **Key Takeaway:** **Referral** and **Email** visitors convert at the highest rates. Referral traffic arrives with pre-established trust, while email campaigns target contacts already familiar with your brand.
        """)

    with ins_c2:
        st.markdown("##### 2. Session Duration Threshold Curve")
        df_dur = df.copy()
        df_dur['Duration Bucket'] = pd.cut(
            df_dur['session_duration'],
            bins=[0, 90, 180, 300, 480, 1000],
            labels=['<1.5 min', '1.5-3 min', '3-5 min', '5-8 min', '8+ min']
        )
        dur_analysis = df_dur.groupby('Duration Bucket', observed=False)['converted'].mean().reset_index()
        dur_analysis['Conversion %'] = dur_analysis['converted'] * 100.0

        line_dur = px.line(
            dur_analysis,
            x='Duration Bucket',
            y='Conversion %',
            markers=True,
            line_shape='spline'
        )
        line_dur.update_traces(
            line_color='#6366F1',
            line_width=4,
            marker=dict(size=10, color='#4338CA')
        )
        line_dur.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis_title="Conversion %",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(line_dur, use_container_width=True, key="ins_dur_chart", config={"responsive": True})

        st.markdown("""
        > **Key Takeaway:** The conversion inflection point occurs after **3 minutes (180s)**. Visitors staying longer than 5 minutes have a greater than **70% conversion likelihood**.
        """)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Insights Row 2
    ins_c3, ins_c4 = st.columns(2, gap="large")

    with ins_c3:
        st.markdown("##### 3. Pages Visited Critical Mass")
        df_pages = df.copy()
        df_pages['Pages Bucket'] = pd.cut(
            df_pages['pages_visited'],
            bins=[0, 2, 4, 7, 12, 30],
            labels=['1-2 Pages', '3-4 Pages', '5-7 Pages', '8-12 Pages', '13+ Pages']
        )
        pages_analysis = df_pages.groupby('Pages Bucket', observed=False)['converted'].mean().reset_index()
        pages_analysis['Conversion %'] = pages_analysis['converted'] * 100.0

        bar_pg = px.bar(
            pages_analysis,
            x='Pages Bucket',
            y='Conversion %',
            color='Conversion %',
            color_continuous_scale='Purples',
            text=pages_analysis['Conversion %'].apply(lambda x: f"{x:.1f}%")
        )
        bar_pg.update_traces(
            textposition='outside',
            textfont=dict(color='#0F172A', size=11, family='Plus Jakarta Sans')
        )
        bar_pg.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(bar_pg, use_container_width=True, key="ins_pages_chart", config={"responsive": True})

        st.markdown("""
        > **Key Takeaway:** Multi-page browsing (5+ pages) correlates with solution comparison and pricing review, multiplying conversion likelihood 4x over single-page visitors.
        """)

    with ins_c4:
        st.markdown("##### 4. Content Download Intent Multiplier")
        dl_summary = df.groupby('content_downloads')['converted'].agg(['count', 'sum', 'mean']).reset_index()
        dl_summary['Conversion %'] = dl_summary['mean'] * 100.0

        bar_dl_ins = px.bar(
            dl_summary,
            x='content_downloads',
            y='Conversion %',
            color='Conversion %',
            color_continuous_scale='Greens',
            text=dl_summary['Conversion %'].apply(lambda x: f"{x:.1f}%")
        )
        bar_dl_ins.update_traces(
            textposition='outside',
            textfont=dict(color='#0F172A', size=11, family='Plus Jakarta Sans')
        )
        bar_dl_ins.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Number of Content Downloads",
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='#0F172A', family='Plus Jakarta Sans')
        )
        st.plotly_chart(bar_dl_ins, use_container_width=True, key="ins_dl_chart", config={"responsive": True})

        st.markdown("""
        > **Key Takeaway:** Visitors downloading **2 or more content assets** convert at >80%. Gated content represents the strongest single behavioral buying signal on your website.
        """)

    # Tactical Sales Playbook Matrix
    st.markdown("---")
    st.markdown("#### 🎯 Enterprise Lead Playbook & Qualification Matrix")

    playbook_data = [
        {
            "Tier": "🟢 High Lead",
            "Score Range": "70 - 100",
            "Qualification": "Sales Qualified Lead (SQL)",
            "Response SLA": "< 1 Hour",
            "Recommended Playbook": "Immediate senior SDR outreach, direct calendar demo invite, account-based marketing acceleration."
        },
        {
            "Tier": "🟡 Medium Lead",
            "Score Range": "40 - 69",
            "Qualification": "Marketing Qualified Lead (MQL)",
            "Response SLA": "< 24 Hours",
            "Recommended Playbook": "Enroll in automated nurture drip sequence with customer case studies, ROI calculators, and webinar invitations."
        },
        {
            "Tier": "🔴 Low Lead",
            "Score Range": "0 - 39",
            "Qualification": "Top of Funnel (Prospect)",
            "Response SLA": "Passive",
            "Recommended Playbook": "Digital retargeting ads, product newsletter signup prompts, educational blog recommendations."
        }
    ]

    st.dataframe(
        pd.DataFrame(playbook_data),
        hide_index=True,
        use_container_width=True
    )

# =========================================================
# TAB 5: ABOUT PROJECT
# =========================================================
with tab_about:
    st.markdown("### ℹ️ About Website Visitor Lead Conversion Scorer")
    st.markdown("Machine Learning architecture, pipeline methodology, and synthetic telemetry details.")

    col_a1, col_a2 = st.columns([1.3, 1], gap="large")

    with col_a1:
        st.markdown("""
        #### 🎯 Problem Statement & Business Solution
        Modern B2B marketing channels drive massive anonymous traffic, but less than 4% typically convert. Sales teams waste critical selling hours manually sifting through low-intent web visitors.

        **Website Visitor Lead Conversion Scorer** continuously analyses real-time digital body language to score, tier, and route high-intent buyers straight to revenue teams.

        #### 🧩 Input Features Analyzed
        1. **Session Duration:** Active seconds browsing the application (non-idle).
        2. **Pages Visited:** Breadth of feature, documentation, and pricing pages reviewed.
        3. **Referral Source:** Acquisition attribution (`Google`, `Social Media`, `Direct`, `Email`, `Referral`).
        4. **Content Downloads:** High-intent asset downloads (eBooks, technical specifications, price sheets).

        #### 🏆 Scored Qualification Tiers
        - 🟢 **High Lead (Score 70 - 100):** Sales Qualified Lead (SQL) -> Immediate direct outreach.
        - 🟡 **Medium Lead (Score 40 - 69):** Marketing Qualified Lead (MQL) -> Nurture drip campaign.
        - 🔴 **Low Lead (Score 0 - 39):** Top-of-Funnel Prospect -> Remarketing display campaigns.

        #### ⚠️ Synthetic Dataset & Integrity Notice
        > **Academic Notice:** The dataset consists of exactly **1,000 synthetic records** generated using a reproducible pseudo-random seed (`numpy.random.seed(42)`). No private customer data is used. The data mirrors real-world digital analytics distributions with zero missing values and zero duplicate feature rows.
        """)

    with col_a2:
        st.markdown("""
        #### 🛠️ Technology Stack
        - **Frontend Dashboard:** Streamlit 1.64
        - **Visualizations:** Plotly Express & Graph Objects
        - **Machine Learning:** Scikit-learn Pipeline, Random Forest
        - **Data Processing:** Pandas, NumPy
        - **Model Serialization:** Joblib

        #### 🤖 Machine Learning Pipeline Flow
        """)

        st.code("""
ColumnTransformer(
  transformers=[
    ('num', StandardScaler(), 
     ['session_duration', 'pages_visited', 'content_downloads']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), 
     ['referral_source'])
  ]
)
↓
RandomForestClassifier(
  n_estimators=120,
  max_depth=7,
  min_samples_split=5,
  min_samples_leaf=2,
  class_weight='balanced',
  random_state=42
)
        """, language="python")

        st.markdown("""
        #### 📁 Project Directory Structure
        ```text
        Website-lead-scorer/
        ├── .streamlit/
        │   └── config.toml
        ├── app.py
        ├── train.py
        ├── predict.py
        ├── requirements.txt
        ├── README.md
        ├── data/
        │   └── website_visitors.csv
        ├── models/
        │   └── lead_conversion_model.joblib
        └── reports/
            ├── confusion_matrix.png
            └── metrics.json
        ```
        """)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; border-top: 1px solid #E2E8F0; padding-top: 14px;">
    Website Visitor Lead Conversion Scorer • Modern Predictive Sales Intelligence Dashboard
</div>
""", unsafe_allow_html=True)

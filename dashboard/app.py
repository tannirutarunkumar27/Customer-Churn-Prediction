"""
dashboard/app.py
----------------
Interactive Streamlit dashboard for visualizing customer churn predictions.

Run with: streamlit run dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px; padding: 1.2rem;
        border: 1px solid #3a3a5c;
    }
    h1, h2, h3 { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
st.sidebar.title("⚙️ Controls")
model_files = [f for f in os.listdir("models") if f.endswith(".joblib")] \
    if os.path.exists("models") else []
selected_model = st.sidebar.selectbox("Select Model", model_files or ["No model found"])

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Monitor churn risk, model performance, and customer insights in one place.")
st.divider()

# ── Load data if available ─────────────────────────────────────────────────────
data_path = os.path.join("data", "processed")
model_loaded = False

if model_files and os.path.exists(os.path.join(data_path, "X_test.npy")):
    model = joblib.load(os.path.join("models", selected_model))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    model_loaded = True

# ── KPI Cards ─────────────────────────────────────────────────────────────────
if model_loaded:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy",  f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("📊 F1 Score",  f"{f1_score(y_test, y_pred):.4f}")
    col3.metric("🔵 ROC-AUC",   f"{roc_auc_score(y_test, y_proba):.4f}")
    col4.metric("🔴 Churn Rate", f"{y_test.mean()*100:.1f}%")
    st.divider()

    # ROC Curve
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC = {roc_auc:.4f}",
                                     line=dict(color="#a78bfa", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(dark_mode=True, template="plotly_dark",
                               xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_b:
        st.subheader("Churn Probability Distribution")
        fig_hist = px.histogram(x=y_proba, color=y_test.astype(str),
                                nbins=40, barmode="overlay",
                                labels={"x": "Churn Probability", "color": "Actual"},
                                color_discrete_sequence=["#6C63FF", "#FF6584"],
                                template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                       labels=dict(x="Predicted", y="Actual"),
                       x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                       template="plotly_dark")
    st.plotly_chart(fig_cm, use_container_width=True)

else:
    st.info("📂 Train a model first and load processed data to see results here.")
    st.markdown("""
    **Steps to get started:**
    1. Place your dataset in `data/raw/churn_data.csv`
    2. Run `python src/data_preprocessing.py`
    3. Run `python src/train.py`
    4. Refresh this dashboard
    """)

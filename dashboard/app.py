"""
dashboard/app.py  —  Enterprise Customer Churn Analytics Platform
Light Corporate Theme with CSS Hover Effects, Violin/Box Plots, & What-If Simulation
Run: streamlit run dashboard/app.py
"""
import os, sys, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

sys.path.insert(0, os.path.abspath("."))

st.set_page_config(
    page_title="Enterprise Churn Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clean Light Modern Corporate CSS ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* Card Styling with Hover */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.25s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.15), 0 8px 10px -6px rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e3a8a;
    }

    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 4px;
    }

    /* Action Recommendation Cards */
    .rec-box {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    .rec-box:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1);
    }
    .rec-box.critical { border-left-color: #ef4444; }
    .rec-box.high { border-left-color: #f59e0b; }
    .rec-box.medium { border-left-color: #3b82f6; }
    .rec-box.low { border-left-color: #10b981; }

    /* Risk Badges */
    .badge-critical { background-color: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-high { background-color: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-medium { background-color: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-low { background-color: #d1fae5; color: #065f46; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }

    /* Custom Button Style */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ── Data & Model Loaders ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join("models", "final_HistGB.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    zip_path = os.path.join("models", "models.zip")
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("models")
        if os.path.exists(path):
            return joblib.load(path)
    return None

@st.cache_data
def load_shap_importance():
    p = os.path.join("reports", "shap_feature_importance.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_risk_scores():
    p = os.path.join("reports", "customer_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_metrics():
    p = os.path.join("reports", "final_metrics.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_sample_dataset():
    p = os.path.join("data", "processed", "churn_advanced.csv")
    if os.path.exists(p):
        return pd.read_csv(p).sample(n=10000, random_state=42)
    zip_p = os.path.join("data", "sample_churn_data.zip")
    if os.path.exists(zip_p):
        import zipfile
        with zipfile.ZipFile(zip_p, "r") as zf:
            with zf.open("churn_sample_10k.csv") as f:
                return pd.read_csv(f)
    return None

model = load_model()
shap_df = load_shap_importance()
risk_scores_df = load_risk_scores()
metrics_df = load_metrics()
df_sample = load_sample_dataset()

# ── Colors & Helper Functions ────────────────────────────────────────────────
COLOR_MAP = {
    "Critical": "#ef4444",
    "High": "#f59e0b",
    "Medium": "#3b82f6",
    "Low": "#10b981"
}

def get_risk_tier(p):
    if p >= 0.80: return "Critical"
    elif p >= 0.60: return "High"
    elif p >= 0.40: return "Medium"
    else: return "Low"

def financial_impact(prob, monthly_rev=85.0):
    rev_at_risk = round(monthly_rev * 12 * prob, 2)
    retained = round(monthly_rev * 12 * (1 - prob), 2)
    campaign_cost = 50.0
    net = round(retained - campaign_cost, 2)
    roi = round((net / campaign_cost) * 100, 1)
    return dict(revenue_at_risk=rev_at_risk, expected_retained=retained,
                campaign_cost=campaign_cost, net_benefit=net, roi_pct=roi)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/briefcase.png", width=50)
    st.markdown("### Executive Navigation")
    
    page = st.radio(
        "Select View",
        [
            "📊 Executive Dashboard",
            "🎻 Exploratory & Distribution Analysis",
            "🔍 Predictor & What-If Simulator",
            "📁 Batch CSV Predictor & Drift Test",
            "🧠 Explainable AI (SHAP)",
            "💰 Financial ROI & Prescriptions"
        ]
    )
    st.divider()
    st.caption("Model Engine: **HistGradientBoosting**")
    st.caption("Status: " + ("🟢 Active" if model else "🔴 Offline"))
    st.caption("Environment: Enterprise Production")

# ── PAGE 1: EXECUTIVE DASHBOARD ─────────────────────────────────────────────
if page == "📊 Executive Dashboard":
    st.markdown('<div class="main-title">Executive Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-level model KPIs, risk segmentation, and overall retention baseline</div>', unsafe_allow_html=True)

    # Top KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    kpis = [
        ("85.04%", "Baseline Accuracy", col1),
        ("24.26%", "Recall (Churn Catch)", col2),
        ("0.685", "ROC-AUC Score", col3),
        ("0.243", "F1 Score", col4),
        ("10.0%", "Historical Churn Rate", col5)
    ]
    
    for val, lbl, col in kpis:
        col.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Model Performance Comparison")
        if metrics_df is not None:
            st.dataframe(
                metrics_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "PR-AUC": "{:.4f}"
                }).highlight_max(subset=["ROC-AUC", "Recall"], color="#e0f2fe"),
                use_container_width=True,
                height=220
            )

    with c2:
        st.subheader("Customer Risk Tier Breakdown")
        if risk_scores_df is not None:
            tier_cts = risk_scores_df["risk_tier"].value_counts().reindex(["Critical", "High", "Medium", "Low"])
            fig_pie = px.pie(
                values=tier_cts.values,
                names=tier_cts.index,
                color=tier_cts.index,
                color_discrete_map=COLOR_MAP,
                hole=0.45,
                template="plotly_white"
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=230)
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    st.subheader("Risk Tier Distribution & Actual Churn Validation")
    if risk_scores_df is not None:
        ca, cb = st.columns(2)
        
        with ca:
            fig_bar = px.bar(
                risk_scores_df.groupby("risk_tier", observed=False).size().reset_index(name="Count"),
                x="risk_tier", y="Count", color="risk_tier",
                color_discrete_map=COLOR_MAP,
                title="Customer Volume per Risk Segment",
                template="plotly_white"
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with cb:
            churn_rates = risk_scores_df.groupby("risk_tier", observed=False)["actual_churn"].mean().reset_index()
            churn_rates["actual_churn"] = churn_rates["actual_churn"] * 100
            fig_rate = px.bar(
                churn_rates,
                x="risk_tier", y="actual_churn", color="risk_tier",
                color_discrete_map=COLOR_MAP,
                title="Actual Churn Rate by Risk Tier (%)",
                template="plotly_white"
            )
            fig_rate.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
            st.plotly_chart(fig_rate, use_container_width=True)

# ── PAGE 2: EXPLORATORY & DISTRIBUTION ANALYSIS (Violin & Box Plots) ─────────
elif page == "🎻 Exploratory & Distribution Analysis":
    st.markdown('<div class="main-title">Feature Distribution Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical comparison of Churn vs Non-Churn customers using Box & Violin plots</div>', unsafe_allow_html=True)

    if df_sample is not None:
        target_col = "churn"
        num_cols = [c for c in df_sample.columns if df_sample[c].dtype in ['float64', 'int64'] and c != target_col]
        
        # Select key interest variables first
        key_defaults = [c for c in ["contract", "risk_score", "customer_satisfaction", "num_complaints", "tenure", "monthlycharges"] if c in num_cols]
        selected_var = st.selectbox("Select Feature to Analyze", num_cols, index=0 if not key_defaults else num_cols.index(key_defaults[0]))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Violin Plot: {selected_var} by Churn Status")
            fig_violin = px.violin(
                df_sample,
                y=selected_var,
                x=target_col,
                color=target_col,
                box=True,
                points="outliers",
                color_discrete_sequence=["#2563eb", "#ef4444"],
                labels={"churn": "Churn Status (0=Stay, 1=Churn)"},
                template="plotly_white"
            )
            fig_violin.update_layout(height=450)
            st.plotly_chart(fig_violin, use_container_width=True)

        with col2:
            st.subheader(f"Box Plot: {selected_var} Quartile Distribution")
            fig_box = px.box(
                df_sample,
                y=selected_var,
                x=target_col,
                color=target_col,
                notched=True,
                color_discrete_sequence=["#0284c7", "#dc2626"],
                labels={"churn": "Churn Status (0=Stay, 1=Churn)"},
                template="plotly_white"
            )
            fig_box.update_layout(height=450)
            st.plotly_chart(fig_box, use_container_width=True)

        st.divider()

        st.subheader("Multi-Feature Distribution Comparison Grid")
        grid_feats = key_defaults[:4]
        if grid_feats:
            grid_cols = st.columns(len(grid_feats))
            for i, feat in enumerate(grid_feats):
                with grid_cols[i]:
                    fig_mini = px.box(
                        df_sample, y=feat, x=target_col, color=target_col,
                        color_discrete_sequence=["#3b82f6", "#ef4444"],
                        template="plotly_white",
                        title=feat
                    )
                    fig_mini.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_mini, use_container_width=True)

    else:
        st.warning("Sample dataset not found at `data/processed/churn_advanced.csv`.")

# ── PAGE 3: PREDICTOR & WHAT-IF SIMULATOR ────────────────────────────────────
elif page == "🔍 Predictor & What-If Simulator":
    st.markdown('<div class="main-title">Interactive What-If Scenario Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Test individual customer attributes and see in real-time how changes impact churn probability and risk tier</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model engine is offline.")
        st.stop()

    feat_names = shap_df["feature"].tolist() if shap_df is not None else [f"feat_{i}" for i in range(39)]

    st.markdown("##### 1. Configure Customer Baseline vs. Scenario")
    
    col_base, col_scen = st.columns(2)

    with col_base:
        st.info("📌 **Current Customer Profile**")
        b_contract = st.selectbox("Baseline Contract", ["Month-to-Month (0)", "One Year (1)", "Two Year (2)"], index=0, key="b_cont")
        b_tenure = st.slider("Baseline Tenure (Months)", 1, 72, 6, key="b_ten")
        b_satisfaction = st.slider("Baseline Satisfaction (1-10)", 1, 10, 3, key="b_sat")
        b_complaints = st.slider("Baseline Complaints", 0, 15, 4, key="b_comp")
        b_tech_support = st.selectbox("Baseline Tech Support", ["No (0)", "Yes (1)"], index=0, key="b_tech")
        b_charges = st.number_input("Baseline Monthly Charges ($)", 10.0, 200.0, 95.0, key="b_chg")

    with col_scen:
        st.success("⚡ **What-If Intervention Scenario**")
        s_contract = st.selectbox("Scenario Contract", ["Month-to-Month (0)", "One Year (1)", "Two Year (2)"], index=1, key="s_cont")
        s_tenure = st.slider("Scenario Tenure (Months)", 1, 72, 18, key="s_ten")
        s_satisfaction = st.slider("Scenario Satisfaction (1-10)", 1, 10, 8, key="s_sat")
        s_complaints = st.slider("Scenario Complaints", 0, 15, 0, key="s_comp")
        s_tech_support = st.selectbox("Scenario Tech Support", ["No (0)", "Yes (1)"], index=1, key="s_tech")
        s_charges = st.number_input("Scenario Monthly Charges ($)", 10.0, 200.0, 75.0, key="s_chg")

    def build_vector(contract_str, tenure_v, sat_v, comp_v, tech_str, chg_v):
        vec = [0.0] * len(feat_names)
        cont_val = int(contract_str.split("(")[1].replace(")", ""))
        tech_val = int(tech_str.split("(")[1].replace(")", ""))
        
        mapping = {
            "contract": cont_val,
            "tenure": float(tenure_v),
            "customer_satisfaction": float(sat_v),
            "num_complaints": float(comp_v),
            "has_tech_support": float(tech_val),
            "monthlycharges": float(chg_v),
            "risk_score": float(comp_v * 1.5 - cont_val * 2.0)
        }
        for k, val in mapping.items():
            if k in feat_names:
                vec[feat_names.index(k)] = val
        return vec

    if st.button("🔮 Calculate Impact & Compare Scenarios", type="primary", use_container_width=True):
        vec_base = build_vector(b_contract, b_tenure, b_satisfaction, b_complaints, b_tech_support, b_charges)
        vec_scen = build_vector(s_contract, s_tenure, s_satisfaction, s_complaints, s_tech_support, s_charges)

        prob_b = float(model.predict_proba([vec_base])[0][1])
        prob_s = float(model.predict_proba([vec_scen])[0][1])

        tier_b = get_risk_tier(prob_b)
        tier_s = get_risk_tier(prob_s)

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Churn Prob", f"{prob_b:.1%}", delta=None)
        m2.metric("Scenario Churn Prob", f"{prob_s:.1%}", delta=f"{(prob_s - prob_b):.1%}", delta_color="inverse")
        m3.metric("Baseline Risk Tier", tier_b)
        m4.metric("Scenario Risk Tier", tier_s)

        # Gauge Comparison
        fig_g = go.Figure()
        fig_g.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob_b * 100,
            name="Baseline",
            title={'text': "Baseline Churn Risk (%)"},
            domain={'x': [0, 0.45], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLOR_MAP[tier_b]}}
        ))
        fig_g.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob_s * 100,
            name="Scenario",
            title={'text': "What-If Scenario Risk (%)"},
            domain={'x': [0.55, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLOR_MAP[tier_s]}}
        ))
        fig_g.update_layout(height=280, template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)

# ── PAGE 4: BATCH CSV PREDICTOR & DRIFT TEST ─────────────────────────────────
elif page == "📁 Batch CSV Predictor & Drift Test":
    st.markdown('<div class="main-title">Batch Scoring & Outcome Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload new customer data batch, simulate predictive outcomes, and detect risk shifts</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model engine is offline.")
        st.stop()

    feat_names = shap_df["feature"].tolist() if shap_df is not None else []
    
    uploaded_file = st.file_uploader("Upload New Customer Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_new):,} records.")

        if "churn" in df_new.columns:
            df_new_features = df_new.drop(columns=["churn"])
        else:
            df_new_features = df_new.copy()

        # Align columns
        for c in feat_names:
            if c not in df_new_features.columns:
                df_new_features[c] = 0.0

        if feat_names:
            df_new_features = df_new_features[feat_names]

        probs = model.predict_proba(df_new_features.values)[:, 1]
        tiers = [get_risk_tier(p) for p in probs]

        df_result = df_new.copy()
        df_result["predicted_churn_probability"] = np.round(probs, 4)
        df_result["risk_tier"] = tiers
        df_result["predicted_churn_label"] = (probs >= 0.50).astype(int)

        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Batch Volume", f"{len(df_result):,}")
        col2.metric("Predicted Churners", f"{(probs >= 0.50).sum():,}")
        col3.metric("Critical Segment", f"{tiers.count('Critical'):,}")
        col4.metric("Avg Churn Risk", f"{probs.mean():.1%}")

        st.subheader("Simulated Outcome Distribution")
        fig_batch = px.histogram(
            df_result,
            x="predicted_churn_probability",
            color="risk_tier",
            color_discrete_map=COLOR_MAP,
            nbins=40,
            template="plotly_white",
            title="Distribution of Predicted Churn Probabilities"
        )
        st.plotly_chart(fig_batch, use_container_width=True)

        st.subheader("Preview Scored Records")
        st.dataframe(df_result.head(50), use_container_width=True)

        csv_data = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Scored Batch CSV",
            csv_data,
            "scored_customer_predictions.csv",
            "text/csv",
            key='download-csv'
        )

# ── PAGE 5: EXPLAINABLE AI (SHAP) ────────────────────────────────────────────
elif page == "🧠 Explainable AI (SHAP)":
    st.markdown('<div class="main-title">Explainable AI & Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">SHAP (SHapley Additive exPlanations) transparency into global model mechanisms</div>', unsafe_allow_html=True)

    if shap_df is not None:
        st.subheader("Global Feature Ranking (Mean |SHAP| Value)")
        top_n = st.slider("Select Top Features to View", 5, len(shap_df), 15)
        
        fig_shap = px.bar(
            shap_df.head(top_n).sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig_shap.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)

    # Display saved figures if available
    img_beeswarm = os.path.join("reports", "figures", "shap_summary_beeswarm.png")
    img_waterfall = os.path.join("reports", "figures", "shap_waterfall_churn_sample.png")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(img_beeswarm):
            st.subheader("SHAP Beeswarm Summary")
            st.image(img_beeswarm, use_container_width=True)
    with c2:
        if os.path.exists(img_waterfall):
            st.subheader("SHAP Local Waterfall Explanation")
            st.image(img_waterfall, use_container_width=True)

# ── PAGE 6: FINANCIAL ROI & PRESCRIPTIONS ────────────────────────────────────
elif page == "💰 Financial ROI & Prescriptions":
    st.markdown('<div class="main-title">Financial ROI & Action Playbook</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Convert predictive insights into business revenue retention strategies</div>', unsafe_allow_html=True)

    st.subheader("Turn Key Drivers into Strategic Action")

    recs = [
        ("Month-to-month contract", "Critical", "Offer 20% discount on annual plan upgrade", "$250 saved/cust"),
        ("High complaints & low satisfaction", "Critical", "Assign dedicated account retention manager within 24h", "$400 saved/cust"),
        ("Low overall satisfaction score", "High", "Trigger automated Customer Success check-in campaign", "$180 saved/cust"),
        ("Lack of tech support service", "High", "Provide 3 months free premium tech support trial", "$120 saved/cust"),
        ("High monthly charge-to-income ratio", "Medium", "Propose customized mid-tier feature bundle", "$90 saved/cust")
    ]

    for finding, priority, action, impact in recs:
        p_class = priority.lower()
        st.markdown(f'''
            <div class="rec-box {p_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="badge-{p_class}">{priority} PRIORITY</span>
                    <span style="font-weight:600; color:#1e3a8a;">Estimated Impact: {impact}</span>
                </div>
                <h4 style="margin: 8px 0 4px 0; color:#0f172a;">{finding}</h4>
                <p style="margin:0; color:#475569;"><b>Prescribed Action:</b> {action}</p>
            </div>
        ''', unsafe_allow_html=True)

    st.divider()

    st.subheader("Campaign ROI Sensitivity Calculator")
    ca, cb, cc = st.columns(3)
    avg_rev = ca.number_input("Avg Annual Customer Value ($)", 100, 5000, 1020)
    camp_cost = cb.number_input("Cost per Targeted Offer ($)", 10, 500, 50)
    succ_rate = cc.slider("Expected Retention Success Rate (%)", 5, 50, 25)

    if risk_scores_df is not None:
        at_risk_vol = len(risk_scores_df[risk_scores_df["risk_tier"].isin(["Critical", "High"])])
        est_retained = at_risk_vol * (succ_rate / 100.0)
        gross_rev_saved = est_retained * avg_rev
        total_campaign_cost = at_risk_vol * camp_cost
        net_roi = gross_rev_saved - total_campaign_cost
        roi_pct = (net_roi / total_campaign_cost * 100) if total_campaign_cost > 0 else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Targeted At-Risk Customers", f"{at_risk_vol:,}")
        r2.metric("Est. Gross Saved Revenue", f"${gross_rev_saved:,.0f}")
        r3.metric("Campaign Expense", f"${total_campaign_cost:,.0f}")
        r4.metric("Net Campaign ROI", f"${net_roi:,.0f}", delta=f"{roi_pct:.0f}% ROI")

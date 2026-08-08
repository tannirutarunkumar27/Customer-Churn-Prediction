# 💼 Enterprise Customer Churn Analytics & Prediction Platform

> **Predict. Explain. Simulate. Retain.**  A production-ready ML platform that identifies at-risk customers, explains *why* they churn using SHAP, simulates *what-if* retention scenarios, scores financial risk, and serves predictions via a FastAPI + Streamlit Light Theme interface.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Light_Theme-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Platform Features](#key-platform-features)
- [Dataset Overview](#dataset-overview)
- [Model Performance Results](#model-performance-results)
- [Violin & Box Plot Exploratory Analysis](#violin--box-plot-exploratory-analysis)
- [What-If Scenario Simulator](#what-if-scenario-simulator)
- [SHAP Explainability](#shap-explainability)
- [Risk Tier Scoring](#risk-tier-scoring)
- [Financial ROI Analysis](#financial-roi-analysis)
- [Prescriptive Recommendations](#prescriptive-recommendations)
- [FastAPI Reference](#fastapi-reference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Setup & Installation](#setup--installation)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Future Work](#future-work)

---

## 🎯 Overview

This platform delivers an end-to-end **Customer Churn Prediction & Analytics** system for telecommunications and subscription-based enterprises. Starting from 1,000,000 raw customer records, it provides:

| Capability | Technical Details |
|-----------|-------------------|
| **Feature Engineering** | 35+ engineered behavior, loyalty, risk, and interaction features |
| **Data Balancing** | Undersampling to 2:1 ratio for optimal minority-class learning |
| **High-Performance Models** | LightGBM, XGBoost, and HistGradientBoosting |
| **Statistical Analysis** | Violin & Box plots for feature distribution comparisons |
| **Scenario Simulation** | Interactive "What-If" simulator comparing Baseline vs. Intervention risk |
| **Explainable AI** | SHAP values (global feature ranking + local waterfall explanations) |
| **Risk Tier Scoring** | 4-tier customer segmentation (Critical, High, Medium, Low) |
| **Financial Impact** | Revenue at risk, campaign expense, net ROI calculation |
| **REST API** | FastAPI with `/predict`, `/batch_predict`, `/health`, `/recommendations` |
| **Light Theme UI** | 6-page Streamlit Dashboard with CSS hover effects and modern styling |
| **Containerization** | Multi-container Docker Compose pipeline |

---

## 🏗️ Architecture

```
Raw Customer Data (1M rows)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature Engineering Pipeline (src/advanced_features.py)    │
│  - 35+ engineered features (interaction, behavior, risk)   │
│  - Target Encoding & Frequency Encoding                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Optimization & Model Engine (src/final_optimization.py)    │
│  - Noise reduction: Filtered 39 low-correlation features    │
│  - RandomUnderSampler: 2:1 class balance                    │
│  - HistGradientBoosting / LightGBM / XGBoost Ensemble        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Explainability & Risk Engine (src/shap_and_risk_scoring.py)│
│  - SHAP TreeExplainer (Beeswarm, Bar, Waterfall plots)      │
│  - 200,000 customer test set scored into 4 risk tiers       │
└────────────┬───────────────────────┬────────────────────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐    ┌──────────────────────────────────┐
    │   FastAPI v2.0  │    │  Streamlit Light Theme Dashboard │
    │  POST /predict  │    │  6 Interactive Views:            │
    │  POST /batch_   │    │  1. Executive Dashboard          │
    │  GET  /health   │    │  2. Violin & Box Plots           │
    │  GET  /recs     │    │  3. What-If Simulator            │
    │  GET  /info     │    │  4. Batch CSV Predictor          │
    └────────┬────────┘    │  5. SHAP Explainability          │
             │             │  6. Financial ROI Playbook       │
             │             └──────────────────┬───────────────┘
             └──────────┬─────────────────────┘
                        ▼
              ┌──────────────────┐
              │  Docker Compose  │
              │  api   :8000      │
              │  dashboard :8501  │
              └──────────────────┘
```

---

## ✨ Key Platform Features

1. **Light Corporate Design System**: Styled with CSS hover cards (`translateY(-4px)`), smooth transitions, custom risk badges, and clean Plotly light templates (`plotly_white`).
2. **Violin & Box Plot Exploratory Analysis**: Allows statistical comparison of numerical feature distributions split by Churn Status (0 vs 1).
3. **What-If Scenario Predictor**: Dual-column interactive simulator comparing a customer's **Baseline Profile** against a proposed **Intervention Offer**.
4. **Batch CSV Predictor & Risk Scoring**: Upload CSV files up to 10,000+ records to score churn probability, assign risk tiers, and export predictions.
5. **SHAP Explainability**: Uncovers global feature importance and individual prediction waterfall breakdowns.
6. **Financial ROI Calculator**: Real-time sensitivity analysis calculating revenue at risk, retention success rate, campaign expenses, and net ROI.

---

## 📊 Dataset Overview

| Attribute | Specification |
|-----------|---------------|
| Dataset Size | 1,000,000 Customer Records |
| Target Variable | `churn` (0 = Stay, 1 = Churn) |
| Class Ratio | 90% Non-Churn / 10% Churn (Severe Imbalance) |
| Training Subset | 238,146 Resampled Records (2:1 Ratio) |
| Test Subset | 200,000 Unseen Test Records |

---

## 🏆 Model Performance Results

| Model | Accuracy | Recall | F1 Score | ROC-AUC | PR-AUC |
|-------|----------|--------|----------|---------|--------|
| **HistGradientBoosting (Default)** | **85.04%** | **24.26%** | **0.243** | **0.685** | **0.205** |
| LightGBM | 84.32% | 25.79% | 0.246 | 0.681 | 0.201 |
| XGBoost | 84.28% | 25.77% | 0.245 | 0.680 | 0.200 |
| **HistGB (Max F1 @ Threshold 0.43)** | 78.77% | **39.49%** | **0.270** | 0.685 | 0.205 |

> **Key Business Insight:** In churn prediction with 90/10 imbalance, predicting "No Churn" yields 90% accuracy but **0% recall**. Our balanced model achieves 85% accuracy while correctly identifying over **24% to 39%** of churning customers.

---

## 🎻 Violin & Box Plot Exploratory Analysis

The dashboard includes a dedicated statistical distribution viewer:
- **Violin Plots**: Visualize feature density and multimodal shapes between Churners and Non-Churners.
- **Box Plots**: Highlight median splits, interquartile ranges (IQR), and extreme outliers.
- **Key Features Analyzed**: `contract`, `risk_score`, `customer_satisfaction`, `num_complaints`, `tenure`, `monthlycharges`.

---

## ⚡ What-If Scenario Simulator

Executives and retention agents can test retention offers before presenting them to customers:

```
[Baseline Customer Profile]              [Proposed Retention Scenario]
Month-to-Month Contract                 Two-Year Contract
Tenure: 6 months          ──►           Tenure: 18 months
Complaints: 4                           Complaints: 0
Tech Support: No                        Tech Support: Yes
Churn Probability: 84.2% (CRITICAL)      Churn Probability: 21.5% (LOW)
```

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) powers model interpretability:

| Rank | Feature | Mean \|SHAP\| Impact | Business Interpretation |
|------|---------|--------------------|-------------------------|
| 1 | `contract` | **0.4392** | Month-to-month contracts are the #1 churn driver |
| 2 | `risk_score` | 0.1249 | High complaints + late payments + service calls |
| 3 | `satisfaction_per_complaint` | 0.1233 | Low satisfaction relative to complaint count |
| 4 | `has_tech_support` | 0.0855 | Absence of tech support increases churn risk |
| 5 | `customer_satisfaction` | 0.0805 | Satisfaction score directly inversely correlates with churn |

---

## 🚨 Risk Tier Scoring

Test set customers (200,000) segmented into 4 operational risk tiers:

| Risk Tier | Probability Range | Volume | Actual Churn Rate | Prescribed Urgency |
|-----------|------------------|--------|-------------------|-------------------|
| 🔴 **Critical** | ≥ 0.80 | 1,437 | **40.78%** | Immediate Outreach (24h) |
| 🟡 **High** | 0.60 – 0.80 | 18,282 | **23.15%** | High Priority (48h) |
| 🔵 **Medium** | 0.40 – 0.60 | 77,286 | **12.19%** | Standard Nurture |
| 🟢 **Low** | < 0.40 | 102,995 | **5.44%** | Low Priority |

---

## 💰 Financial ROI Analysis

**Scenario Assumptions:** Average Customer Value = $1,020/year, Offer Cost = $50/customer, Retention Rate = 25%.

- **At-Risk Target Volume:** 19,719 customers (Critical + High)
- **Gross Revenue at Risk:** ~$20.1 Million
- **Estimated Gross Revenue Saved:** ~$5.0 Million
- **Campaign Expense:** ~$985,950
- **Net Campaign ROI:** **+$4.04 Million (+410% ROI)**

---

## 💡 Prescriptive Recommendations

| Customer Finding | Priority | Recommended Intervention |
|------------------|----------|--------------------------|
| Month-to-month contract | 🔴 Critical | Offer 20% discount on annual plan upgrade |
| High complaints & low satisfaction | 🔴 Critical | Assign dedicated account retention manager |
| Low satisfaction score | 🟡 High | Trigger automated Customer Success check-in |
| No tech support | 🟡 High | Provide 3 months free premium tech support trial |
| High charge-to-income ratio | 🔵 Medium | Propose customized mid-tier feature bundle |

---

## 🔌 FastAPI Reference

**Base URL:** `http://localhost:8000`

- `GET /health`: Health check and model load status
- `GET /model/info`: Model metadata, feature count, and risk bands
- `GET /recommendations`: List of prescriptive business recommendations
- `POST /predict`: Single customer churn prediction, risk tier, and financial ROI
- `POST /batch_predict`: Bulk customer prediction (up to 10,000 records)

---

## 📊 Streamlit Dashboard

Launch the 6-page interactive web dashboard:
```bash
python -m streamlit run dashboard/app.py
```
**Access:** `http://localhost:8501`

---

## ⚙️ Setup & Installation

```bash
# 1. Clone repository
git clone https://github.com/tannirutarunkumar27/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run feature engineering & model training
python src/advanced_features.py
python src/final_optimization.py
python src/shap_and_risk_scoring.py

# 4. Launch API & Dashboard
python -m uvicorn api.main:app --port 8000 &
python -m streamlit run dashboard/app.py --port 8501
```

---

## 🐳 Docker Deployment

Run both the API and Streamlit Dashboard using Docker Compose:

```bash
docker-compose up --build
```
- **API Swagger Docs:** `http://localhost:8000/docs`
- **Streamlit Dashboard:** `http://localhost:8501`

---

## 👤 Author

**Tanniru Tarun Kumar**  
[GitHub](https://github.com/tannirutarunkumar27) · [LinkedIn](https://linkedin.com/in/tannirutarunkumar27)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

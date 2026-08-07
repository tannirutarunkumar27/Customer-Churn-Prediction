# 📉 Customer Churn Prediction

A complete end-to-end machine learning pipeline to predict customer churn using structured data — featuring data preprocessing, exploratory analysis, model training, evaluation, an interactive dashboard, and a REST API.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/              # Original, immutable raw datasets
│   └── processed/        # Cleaned and feature-engineered datasets
│
├── notebooks/            # Jupyter notebooks for EDA and experimentation
│
├── src/                  # Core Python source modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/               # Saved model files (.pkl, .joblib)
│
├── reports/              # Generated analysis reports, plots, and metrics
│
├── dashboard/            # Streamlit interactive dashboard
│   └── app.py
│
├── api/                  # FastAPI REST API for model serving
│   └── main.py
│
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your Dataset
Place your raw dataset (e.g., `churn_data.csv`) in the `data/raw/` folder.

---

## 🔄 Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/data_preprocessing.py` | Load, clean, encode, and split data |
| 2 | `src/feature_engineering.py` | Create new features, scaling |
| 3 | `src/train.py` | Train ML models (XGBoost, LightGBM, etc.) |
| 4 | `src/evaluate.py` | Evaluate models, generate metrics & plots |

---

## 📊 Running the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🌐 Running the API

```bash
uvicorn api.main:app --reload
```

API docs available at: `http://127.0.0.1:8000/docs`

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📈 Models Used

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM

---

## 📋 Reports

All generated plots, confusion matrices, ROC curves, and metrics are saved in the `reports/` directory.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn**, **XGBoost**, **LightGBM**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**, **Plotly**
- **Streamlit** (dashboard)
- **FastAPI** (REST API)

---

## 📄 License

MIT License — feel free to use and modify.

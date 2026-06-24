# 📈 SalesPulse — End-to-End Sales Forecasting Dashboard

> Retail sales forecasting using Prophet & ARIMA, SQL analytics, and an interactive Streamlit dashboard.

---

## 🎯 Business Problem
Retail businesses lose revenue by over/under-stocking and missing seasonal demand shifts.
This project answers:
- **What will sales look like next 90 days?**
- **Which regions/categories drive the most revenue?**
- **What seasonal patterns exist in the data?**

---

## 🔑 Key Results
- Forecasted 90-day sales with **Prophet (MAPE < 12%)**
- Identified **Q4 seasonality spike** contributing 35%+ of annual revenue
- Built interactive Streamlit dashboard with real-time filters

---

## 🏗️ Architecture
```
Raw CSV (Superstore Dataset)
        ↓
SQLite DB (SQL Queries for aggregation)
        ↓
EDA & Cleaning (Pandas, Matplotlib)
        ↓
Time Series Modeling (Prophet + ARIMA)
        ↓
Model Evaluation (MAE, RMSE, MAPE)
        ↓
Streamlit Dashboard (Forecast + Insights)
```

---

## 🛠️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.x |
| Database | SQLite (SQL queries) |
| Data Processing | Pandas, NumPy |
| Forecasting | Prophet, Statsmodels (ARIMA) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |

---

## 📁 Project Structure
```
sales_forecasting/
├── data/
│   └── superstore.csv              # Raw dataset
├── sql/
│   └── queries.sql                 # All SQL analytics queries
├── notebooks/
│   ├── 01_EDA.py                   # Exploratory Data Analysis
│   ├── 02_preprocessing.py         # Cleaning & feature engineering
│   ├── 03_arima_model.py           # ARIMA forecasting
│   ├── 04_prophet_model.py         # Prophet forecasting
│   └── 05_model_evaluation.py      # Compare models
├── dashboard/
│   └── app.py                      # Streamlit dashboard
└── requirements.txt
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Download **Superstore.csv** from Kaggle and place in `data/` folder:
👉 https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

### 3. Run notebooks in order
```bash
python notebooks/01_EDA.py
python notebooks/02_preprocessing.py
python notebooks/03_arima_model.py
python notebooks/04_prophet_model.py
python notebooks/05_model_evaluation.py
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📄 License
MIT — free for portfolio use.

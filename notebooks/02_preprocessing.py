# ============================================================
# 02_preprocessing.py — Data Cleaning & Feature Engineering
# SalesPulse: End-to-End Sales Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Load & Clean ──────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading & Cleaning Data")
print("=" * 55)

df = pd.read_csv(r"D:\PROJECTS\DA+DS\sales_forecasting\data\Sample - Superstore.csv", encoding="latin-1")
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(" ", "_").str.replace("-", "_"))
df["order_date"] = pd.to_datetime(df["order_date"])
df["ship_date"]  = pd.to_datetime(df["ship_date"])

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped {before - len(df)} duplicates. Remaining: {len(df):,} rows")

# Derived columns
df["ship_days"]     = (df["ship_date"] - df["order_date"]).dt.days
df["profit_margin"] = df["profit"] / df["sales"].replace(0, np.nan)
df["year"]          = df["order_date"].dt.year
df["month"]         = df["order_date"].dt.month
df["quarter"]       = df["order_date"].dt.quarter
df["day_of_week"]   = df["order_date"].dt.dayofweek  # 0=Mon

print(f"\nNew features added: ship_days, profit_margin, year, month, quarter, day_of_week")

# ── 2. Build Time Series ─────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Building Daily & Monthly Time Series")
print("=" * 55)

# Daily aggregation
daily = (df.groupby("order_date")
           .agg(sales=("sales", "sum"),
                orders=("order_id", "nunique"),
                profit=("profit", "sum"))
           .reset_index()
           .rename(columns={"order_date": "ds"}))

# Fill any missing dates with 0
full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
daily = (daily.set_index("ds")
              .reindex(full_range, fill_value=0)
              .rename_axis("ds")
              .reset_index())

# Monthly aggregation
monthly = (df.set_index("order_date")
             .resample("MS")["sales"]
             .sum()
             .reset_index()
             .rename(columns={"order_date": "ds", "sales": "y"}))

print(f"Daily series  : {len(daily):,} rows ({daily['ds'].min().date()} → {daily['ds'].max().date()})")
print(f"Monthly series: {len(monthly)} rows")
print(f"\nMonthly preview:\n{monthly.tail(5).to_string(index=False)}")

# ── 3. Stationarity Test ─────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: ADF Stationarity Test (Monthly Series)")
print("=" * 55)

result = adfuller(monthly["y"])
print(f"  ADF Statistic : {result[0]:.4f}")
print(f"  p-value       : {result[1]:.4f}")
print(f"  Conclusion    : {'✅ Stationary (p < 0.05)' if result[1] < 0.05 else '⚠️  Non-stationary — differencing needed for ARIMA'}")

# Differenced series
monthly["y_diff"] = monthly["y"].diff()
result_diff = adfuller(monthly["y_diff"].dropna())
print(f"\nAfter 1st difference:")
print(f"  ADF Statistic : {result_diff[0]:.4f}")
print(f"  p-value       : {result_diff[1]:.4f}")
print(f"  Conclusion    : {'✅ Stationary' if result_diff[1] < 0.05 else '⚠️  Still non-stationary'}")

# ── 4. Rolling Statistics Plot ───────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Rolling Mean & Std Plot")
print("=" * 55)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("SalesPulse — Time Series Stationarity Analysis", fontweight="bold")

ts = monthly.set_index("ds")["y"]
rolling_mean = ts.rolling(3).mean()
rolling_std  = ts.rolling(3).std()

axes[0].plot(ts.index, ts.values, label="Monthly Sales", color="#2563EB", linewidth=1.5)
axes[0].plot(rolling_mean.index, rolling_mean.values, label="3-Month Rolling Mean",
             color="#DC2626", linewidth=2, linestyle="--")
axes[0].fill_between(rolling_std.index,
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.15, color="#DC2626")
axes[0].set_title("Monthly Sales with Rolling Statistics")
axes[0].set_ylabel("Sales ($)")
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

axes[1].plot(monthly["ds"].iloc[1:], monthly["y_diff"].iloc[1:],
             color="#16A34A", linewidth=1.5, label="1st Difference")
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("1st Differenced Series")
axes[1].set_ylabel("Δ Sales ($)")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/02_stationarity.png", dpi=150, bbox_inches="tight")
print("✅ Plot saved → outputs/02_stationarity.png")
plt.show()

# ── 5. Save Processed Data ───────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Saving Processed Datasets")
print("=" * 55)

daily.to_csv("data/daily_sales.csv", index=False)
monthly.to_csv("data/monthly_sales.csv", index=False)
df.to_csv("data/superstore_clean.csv", index=False)

print("✅ Saved: data/daily_sales.csv")
print("✅ Saved: data/monthly_sales.csv")
print("✅ Saved: data/superstore_clean.csv")
print("\n✅ Preprocessing complete. Run 03_arima_model.py next.\n")

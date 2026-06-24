# ============================================================
# 03_arima_model.py — ARIMA Time Series Forecasting
# SalesPulse: End-to-End Sales Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── 1. Load Monthly Series ───────────────────────────────────
print("=" * 55)
print("ARIMA FORECASTING MODEL")
print("=" * 55)

monthly = pd.read_csv("data/monthly_sales.csv", parse_dates=["ds"])
monthly = monthly.set_index("ds")["y"]

print(f"Series length: {len(monthly)} months")
print(f"Range: {monthly.index.min().date()} → {monthly.index.max().date()}")

# ── 2. ACF & PACF Plots (to pick p, d, q) ───────────────────
print("\nPlotting ACF & PACF to determine ARIMA order...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ACF & PACF — ARIMA Order Selection", fontweight="bold")
plot_acf(monthly.diff().dropna(),  ax=axes[0], lags=20, title="Autocorrelation (ACF)")
plot_pacf(monthly.diff().dropna(), ax=axes[1], lags=20, title="Partial Autocorrelation (PACF)")
plt.tight_layout()
plt.savefig("outputs/03_acf_pacf.png", dpi=150, bbox_inches="tight")
print("✅ Saved → outputs/03_acf_pacf.png")
plt.show()

# ── 3. Train / Test Split ────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Train/Test Split (last 6 months = test)")
print("=" * 55)

test_size  = 6
train      = monthly[:-test_size]
test       = monthly[-test_size:]

print(f"Train: {len(train)} months | Test: {len(test)} months")

# ── 4. Fit ARIMA ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Fitting ARIMA(2,1,2)")
print("=" * 55)

# order=(p,d,q): p=2 (AR lags), d=1 (1 difference), q=2 (MA lags)
model = ARIMA(train, order=(2, 1, 2))
fitted = model.fit()

print(fitted.summary())

# ── 5. Forecast ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Forecasting Test Period + 3 Future Months")
print("=" * 55)

# In-sample forecast for test period
forecast_test  = fitted.forecast(steps=test_size)

# Retrain on full data, then forecast 3 future months
model_full     = ARIMA(monthly, order=(2, 1, 2))
fitted_full    = model_full.fit()
forecast_future = fitted_full.get_forecast(steps=3)
forecast_mean  = forecast_future.predicted_mean
conf_int       = forecast_future.conf_int()

print(f"\n📅 3-Month Future Forecast:")
for date, val, lo, hi in zip(
        forecast_mean.index,
        forecast_mean.values,
        conf_int.iloc[:, 0].values,
        conf_int.iloc[:, 1].values):
    print(f"  {date.strftime('%b %Y')} → ${val:>10,.2f}  (CI: ${lo:,.0f} – ${hi:,.0f})")

# ── 6. Evaluation Metrics ────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Model Evaluation on Test Set")
print("=" * 55)

mae  = mean_absolute_error(test, forecast_test)
rmse = np.sqrt(mean_squared_error(test, forecast_test))
mape = np.mean(np.abs((test.values - forecast_test.values) / test.values)) * 100

print(f"  MAE  : ${mae:>10,.2f}")
print(f"  RMSE : ${rmse:>10,.2f}")
print(f"  MAPE : {mape:>9.2f}%")

# ── 7. Visualization ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title("ARIMA(2,1,2) — Sales Forecast vs Actual", fontsize=14, fontweight="bold")

ax.plot(train.index, train.values, label="Train Data", color="#2563EB", linewidth=1.8)
ax.plot(test.index,  test.values,  label="Actual (Test)", color="#16A34A",
        linewidth=2, marker="o", markersize=5)
ax.plot(test.index,  forecast_test.values, label="ARIMA Forecast (Test)",
        color="#DC2626", linewidth=2, linestyle="--", marker="x", markersize=6)
ax.plot(forecast_mean.index, forecast_mean.values,
        label="Future Forecast", color="#F59E0B", linewidth=2.5, marker="^", markersize=7)
ax.fill_between(conf_int.index,
                conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                alpha=0.2, color="#F59E0B", label="95% CI")

ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1.5, label="Train/Test Split")
ax.set_ylabel("Monthly Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/03_arima_forecast.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved → outputs/03_arima_forecast.png")
plt.show()

# ── 8. Save Results ──────────────────────────────────────────
results = pd.DataFrame({
    "ds"          : list(test.index) + list(forecast_mean.index),
    "actual"      : list(test.values) + [np.nan] * 3,
    "arima_pred"  : list(forecast_test.values) + list(forecast_mean.values),
    "type"        : ["test"] * test_size + ["future"] * 3
})
results.to_csv("data/arima_results.csv", index=False)
print("✅ Saved → data/arima_results.csv")
print("\n✅ ARIMA complete. Run 04_prophet_model.py next.\n")

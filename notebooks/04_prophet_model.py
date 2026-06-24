# ============================================================
# 04_prophet_model.py — Prophet Time Series Forecasting
# SalesPulse: End-to-End Sales Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 55)
print("PROPHET FORECASTING MODEL")
print("=" * 55)

monthly = pd.read_csv("data/monthly_sales.csv", parse_dates=["ds"])
# Prophet requires columns named exactly 'ds' and 'y'
monthly = monthly[["ds", "y"]].copy()

print(f"Series: {len(monthly)} monthly records")
print(f"Range : {monthly['ds'].min().date()} → {monthly['ds'].max().date()}")

# ── 2. Train / Test Split ────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Train/Test Split (last 6 months = test)")
print("=" * 55)

test_size = 6
train = monthly.iloc[:-test_size]
test  = monthly.iloc[-test_size:]

print(f"Train: {len(train)} | Test: {len(test)}")

# ── 3. Build & Fit Prophet ───────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Fitting Prophet Model")
print("=" * 55)

model = Prophet(
    yearly_seasonality  = True,
    weekly_seasonality  = False,   # monthly data — no weekly pattern
    daily_seasonality   = False,
    seasonality_mode    = "multiplicative",   # good for growing series
    changepoint_prior_scale = 0.05,           # flexibility of trend
    interval_width      = 0.95
)

# Add US public holidays (optional boost for retail)
model.add_country_holidays(country_name="US")

model.fit(train)
print("✅ Prophet model fitted successfully.")

# ── 4. Forecast ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Forecasting (Test + 3 Future Months)")
print("=" * 55)

# Create future dataframe: covers test period + 3 extra months
future = model.make_future_dataframe(periods=test_size + 3, freq="MS")
forecast = model.predict(future)

# Split forecast output
test_forecast   = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]]
future_forecast = forecast[~forecast["ds"].isin(monthly["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]]

print(f"\n📅 3-Month Future Forecast:")
for _, row in future_forecast.iterrows():
    print(f"  {row['ds'].strftime('%b %Y')} → ${row['yhat']:>10,.2f}  "
          f"(CI: ${row['yhat_lower']:,.0f} – ${row['yhat_upper']:,.0f})")

# ── 5. Metrics ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Model Evaluation")
print("=" * 55)

y_true = test["y"].values
y_pred = test_forecast["yhat"].values

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"  MAE  : ${mae:>10,.2f}")
print(f"  RMSE : ${rmse:>10,.2f}")
print(f"  MAPE : {mape:>9.2f}%")

# ── 6. Prophet Component Plots ───────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Saving Prophet Component Plots")
print("=" * 55)

fig_comp = model.plot_components(forecast)
fig_comp.suptitle("Prophet — Trend & Seasonal Components", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/04_prophet_components.png", dpi=150, bbox_inches="tight")
print("✅ Saved → outputs/04_prophet_components.png")
plt.show()

# ── 7. Custom Forecast Chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title("Prophet — Sales Forecast vs Actual", fontsize=14, fontweight="bold")

# Training actuals
ax.plot(train["ds"], train["y"], label="Train Data", color="#2563EB", linewidth=1.8)

# Test actuals
ax.plot(test["ds"], test["y"], label="Actual (Test)",
        color="#16A34A", linewidth=2, marker="o", markersize=6)

# Test forecast with CI
ax.plot(test_forecast["ds"], test_forecast["yhat"],
        label="Prophet Forecast (Test)", color="#DC2626",
        linewidth=2, linestyle="--", marker="x", markersize=7)
ax.fill_between(test_forecast["ds"],
                test_forecast["yhat_lower"], test_forecast["yhat_upper"],
                alpha=0.15, color="#DC2626")

# Future forecast
ax.plot(future_forecast["ds"], future_forecast["yhat"],
        label="Future Forecast (3M)", color="#F59E0B",
        linewidth=2.5, marker="^", markersize=8)
ax.fill_between(future_forecast["ds"],
                future_forecast["yhat_lower"], future_forecast["yhat_upper"],
                alpha=0.2, color="#F59E0B", label="95% CI (Future)")

ax.axvline(test["ds"].iloc[0], color="gray", linestyle=":", linewidth=1.5, label="Train/Test Split")
ax.set_ylabel("Monthly Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/04_prophet_forecast.png", dpi=150, bbox_inches="tight")
print("✅ Saved → outputs/04_prophet_forecast.png")
plt.show()

# ── 8. Save Results ──────────────────────────────────────────
prophet_results = pd.concat([
    test_forecast.assign(actual=y_true, type="test"),
    future_forecast.assign(actual=np.nan, type="future")
])
prophet_results.to_csv("data/prophet_results.csv", index=False)
print("✅ Saved → data/prophet_results.csv")
print("\n✅ Prophet complete. Run 05_model_evaluation.py next.\n")

# ============================================================
# 05_model_evaluation.py — Compare ARIMA vs Prophet
# SalesPulse: End-to-End Sales Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Results ──────────────────────────────────────────
print("=" * 55)
print("MODEL COMPARISON: ARIMA vs Prophet")
print("=" * 55)

monthly      = pd.read_csv("data/monthly_sales.csv", parse_dates=["ds"])
arima_res    = pd.read_csv("data/arima_results.csv", parse_dates=["ds"])
prophet_res  = pd.read_csv("data/prophet_results.csv", parse_dates=["ds"])

# Test period only
arima_test   = arima_res[arima_res["type"] == "test"]
prophet_test = prophet_res[prophet_res["type"] == "test"]

# Align on same test dates
test_actual  = monthly[monthly["ds"].isin(arima_test["ds"])]["y"].values

arima_preds  = arima_test["arima_pred"].values
prophet_preds = prophet_test["yhat"].values

# ── 2. Metrics Comparison ────────────────────────────────────
print("\n" + "=" * 55)
print("METRICS COMPARISON (Test Period)")
print("=" * 55)

def compute_metrics(actual, predicted, name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

metrics = pd.DataFrame([
    compute_metrics(test_actual, arima_preds,   "ARIMA(2,1,2)"),
    compute_metrics(test_actual, prophet_preds, "Prophet"),
])

metrics_display = metrics.copy()
metrics_display["MAE"]      = metrics_display["MAE"].map("${:,.2f}".format)
metrics_display["RMSE"]     = metrics_display["RMSE"].map("${:,.2f}".format)
metrics_display["MAPE (%)"] = metrics_display["MAPE (%)"].map("{:.2f}%".format)
print(metrics_display.to_string(index=False))

winner = metrics.loc[metrics["MAPE (%)"].idxmin(), "Model"]
print(f"\n🏆 Best Model: {winner} (lower MAPE)")

# ── 3. Side-by-Side Forecast Chart ───────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Side-by-Side Forecast Comparison Chart")
print("=" * 55)

arima_future   = arima_res[arima_res["type"]   == "future"]
prophet_future = prophet_res[prophet_res["type"] == "future"]

fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
fig.suptitle("ARIMA vs Prophet — Forecast Comparison", fontsize=14, fontweight="bold")

for ax, model_name, preds, future_ds, future_vals in [
    (axes[0], "ARIMA(2,1,2)",
     arima_preds, arima_future["ds"], arima_future["arima_pred"]),
    (axes[1], "Prophet",
     prophet_preds, prophet_future["ds"], prophet_future["yhat"]),
]:
    ax.plot(monthly["ds"], monthly["y"], color="#2563EB",
            linewidth=1.8, label="Historical")
    ax.plot(arima_test["ds"], test_actual, color="#16A34A",
            linewidth=2, marker="o", markersize=5, label="Actual (Test)")
    ax.plot(arima_test["ds"], preds, color="#DC2626",
            linewidth=2, linestyle="--", marker="x", label=f"{model_name} (Test)")
    ax.plot(future_ds, future_vals, color="#F59E0B",
            linewidth=2.5, marker="^", markersize=7, label="Future (3M)")
    ax.axvline(arima_test["ds"].iloc[0], color="gray",
               linestyle=":", linewidth=1.5)
    ax.set_title(model_name, fontweight="bold")
    ax.set_ylabel("Monthly Sales ($)" if ax == axes[0] else "")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/05_model_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Saved → outputs/05_model_comparison.png")
plt.show()

# ── 4. Metrics Bar Chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Model Performance Comparison", fontweight="bold")
metric_cols = ["MAE", "RMSE", "MAPE (%)"]
colors = ["#2563EB", "#DC2626"]

for ax, col in zip(axes, metric_cols):
    bars = ax.bar(metrics["Model"], metrics[col], color=colors, width=0.5)
    ax.set_title(col, fontweight="bold")
    ax.set_ylabel(col)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.02,
                f"{bar.get_height():,.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/05_metrics_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Saved → outputs/05_metrics_comparison.png")
plt.show()

# ── 5. Save Best Forecast for Dashboard ──────────────────────
print("\n" + "=" * 55)
print("STEP 5: Saving Final Forecast (Best Model)")
print("=" * 55)

# Use Prophet results as final (typically better for retail seasonality)
final = pd.concat([
    monthly.rename(columns={"y": "actual"}).assign(predicted=np.nan, type="historical"),
    prophet_test[["ds", "yhat", "yhat_lower", "yhat_upper"]].assign(
        actual=test_actual, type="test"
    ).rename(columns={"yhat": "predicted"}),
    prophet_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].assign(
        actual=np.nan, type="future"
    ).rename(columns={"yhat": "predicted"}),
])

final.to_csv("data/final_forecast.csv", index=False)
print(f"✅ Saved → data/final_forecast.csv ({len(final)} rows)")

print(f"""
╔══════════════════════════════════════════╗
║       MODEL EVALUATION SUMMARY          ║
╠══════════════════════════════════════════╣
║  Best Model : {winner:<27}║
║  MAPE       : {metrics.loc[metrics['Model']==winner,'MAPE (%)'].values[0]:<27.2f}║
║  Historical : {len(monthly):<27} months ║
║  Forecasted : 3 future months            ║
╚══════════════════════════════════════════╝
""")
print("✅ All done! Launch the dashboard with:\n   streamlit run dashboard/app.py\n")

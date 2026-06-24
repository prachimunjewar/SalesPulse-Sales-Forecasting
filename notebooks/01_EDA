# ============================================================
# 01_EDA.py — Exploratory Data Analysis
# SalesPulse: End-to-End Sales Forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sqlite3
import os

# ── Setup ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading Dataset")
print("=" * 55)

df = pd.read_csv(r"data\Sample - Superstore.csv", encoding="latin-1")

# Standardize column names
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("-", "_"))

print(f"Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns   : {list(df.columns)}\n")
print(df.head(3).to_string())

# ── 2. Data Types & Nulls ────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Data Types & Missing Values")
print("=" * 55)

# Fix dates
df["order_date"] = pd.to_datetime(df["order_date"])
df["ship_date"]  = pd.to_datetime(df["ship_date"])

null_summary = pd.DataFrame({
    "dtype"  : df.dtypes,
    "nulls"  : df.isnull().sum(),
    "null_%" : (df.isnull().mean() * 100).round(2)
})
print(null_summary[null_summary["nulls"] > 0] if null_summary["nulls"].any()
      else "✅ No missing values found.")

print(f"\nDate range: {df['order_date'].min().date()} → {df['order_date'].max().date()}")

# ── 3. Key Stats ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Business KPIs")
print("=" * 55)

print(f"  Total Revenue    : ${df['sales'].sum():>12,.2f}")
print(f"  Total Profit     : ${df['profit'].sum():>12,.2f}")
print(f"  Profit Margin    : {df['profit'].sum()/df['sales'].sum()*100:>11.2f}%")
print(f"  Unique Customers : {df['customer_id'].nunique():>12,}")
print(f"  Total Orders     : {df['order_id'].nunique():>12,}")
print(f"  Avg Order Value  : ${df.groupby('order_id')['sales'].sum().mean():>12,.2f}")

# ── 4. Load into SQLite & run SQL queries ────────────────────
print("\n" + "=" * 55)
print("STEP 4: Loading into SQLite & Running SQL Analytics")
print("=" * 55)

conn = sqlite3.connect("data/superstore.db")
df.to_sql("superstore", conn, if_exists="replace", index=False)
print("✅ Data loaded into SQLite (data/superstore.db)")

# Monthly trend via SQL
monthly_sql = pd.read_sql("""
    SELECT strftime('%Y-%m', order_date) AS month,
           ROUND(SUM(sales), 2)          AS monthly_sales,
           COUNT(DISTINCT order_id)      AS order_count
    FROM superstore
    GROUP BY month
    ORDER BY month
""", conn)

regional_sql = pd.read_sql("""
    SELECT region,
           ROUND(SUM(sales), 2)  AS total_sales,
           ROUND(SUM(profit), 2) AS total_profit
    FROM superstore
    GROUP BY region
    ORDER BY total_sales DESC
""", conn)

print("\nMonthly Sales (SQL preview):")
print(monthly_sql.tail(5).to_string(index=False))

print("\nRegional Sales (SQL):")
print(regional_sql.to_string(index=False))
conn.close()

# ── 5. Visualizations ────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Generating EDA Plots")
print("=" * 55)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("SalesPulse — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

# (A) Monthly revenue trend
monthly_sql["month_dt"] = pd.to_datetime(monthly_sql["month"])
ax = axes[0, 0]
ax.plot(monthly_sql["month_dt"], monthly_sql["monthly_sales"], color="#2563EB", linewidth=2)
ax.fill_between(monthly_sql["month_dt"], monthly_sql["monthly_sales"], alpha=0.15, color="#2563EB")
ax.set_title("Monthly Sales Trend", fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# (B) Sales by Category
cat_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)
ax = axes[0, 1]
bars = ax.bar(cat_sales.index, cat_sales.values, color=["#2563EB", "#16A34A", "#DC2626"])
ax.set_title("Sales by Category", fontweight="bold")
ax.set_ylabel("Total Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)

# (C) Regional Sales
ax = axes[0, 2]
colors = ["#2563EB", "#16A34A", "#F59E0B", "#DC2626"]
ax.bar(regional_sql["region"], regional_sql["total_sales"], color=colors)
ax.set_title("Sales by Region", fontweight="bold")
ax.set_ylabel("Total Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# (D) Quarterly Seasonality
df["quarter"] = df["order_date"].dt.to_period("Q").astype(str)
quarterly = df.groupby("quarter")["sales"].sum()
ax = axes[1, 0]
ax.plot(range(len(quarterly)), quarterly.values, marker="o", color="#7C3AED", linewidth=2)
ax.set_title("Quarterly Sales (Seasonality)", fontweight="bold")
ax.set_xticks(range(len(quarterly)))
ax.set_xticklabels(quarterly.index, rotation=45, fontsize=7)
ax.set_ylabel("Sales ($)")

# (E) Discount vs Profit scatter
ax = axes[1, 1]
sample = df.sample(min(2000, len(df)), random_state=42)
sc = ax.scatter(sample["discount"], sample["profit"],
                alpha=0.4, c=sample["sales"], cmap="Blues", s=15)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_title("Discount vs Profit", fontweight="bold")
ax.set_xlabel("Discount Rate"); ax.set_ylabel("Profit ($)")
plt.colorbar(sc, ax=ax, label="Sales")

# (F) Top 10 Sub-categories
sub_sales = df.groupby("sub_category")["sales"].sum().nlargest(10)
ax = axes[1, 2]
ax.barh(sub_sales.index[::-1], sub_sales.values[::-1], color="#2563EB")
ax.set_title("Top 10 Sub-Categories by Sales", fontweight="bold")
ax.set_xlabel("Total Sales ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("outputs/01_EDA_plots.png", dpi=150, bbox_inches="tight")
print("✅ EDA plots saved → outputs/01_EDA_plots.png")
plt.show()

print("\n✅ EDA complete. Run 02_preprocessing.py next.\n")

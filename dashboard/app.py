
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SalesPulse AI Dashboard", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("superstore_clean.csv", parse_dates=["order_date","ship_date"])
    forecast = pd.read_csv("final_forecast.csv", parse_dates=["ds"])
    monthly = pd.read_csv("monthly_sales.csv", parse_dates=["ds"])
    return df, forecast, monthly

df, forecast, monthly = load_data()

st.title("📈 SalesPulse AI Forecasting Dashboard")
st.caption("Executive Analytics • Forecasting • Customer Intelligence")

# Sidebar
st.sidebar.header("Filters")

region = st.sidebar.selectbox("Region", ["All"] + sorted(df["region"].dropna().unique().tolist()))
category = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique().tolist()))
segment = st.sidebar.selectbox("Segment", ["All"] + sorted(df["segment"].dropna().unique().tolist()))

filtered = df.copy()

if region != "All":
    filtered = filtered[filtered["region"] == region]

if category != "All":
    filtered = filtered[filtered["category"] == category]

if segment != "All":
    filtered = filtered[filtered["segment"] == segment]

revenue = filtered["sales"].sum()
profit = filtered["profit"].sum()
margin = (profit / revenue * 100) if revenue else 0
orders = filtered["order_id"].nunique()
customers = filtered["customer_id"].nunique()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Dashboard", "Forecasting", "Customer Analytics", "Data Explorer"]
)

with tab1:

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Revenue", f"${revenue:,.0f}")
    c2.metric("Profit", f"${profit:,.0f}")
    c3.metric("Margin", f"{margin:.2f}%")
    c4.metric("Orders", f"{orders:,}")
    c5.metric("Customers", f"{customers:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        yearly = filtered.groupby("year")["sales"].sum().reset_index()

        fig = px.line(
            yearly,
            x="year",
            y="sales",
            markers=True,
            title="Revenue Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_sales = (
            filtered.groupby("region")["sales"]
            .sum()
            .reset_index()
            .sort_values("sales")
        )

        fig2 = px.bar(
            region_sales,
            x="sales",
            y="region",
            orientation="h",
            title="Sales by Region"
        )

        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        cat = (
            filtered.groupby("category")["sales"]
            .sum()
            .reset_index()
        )

        fig3 = px.pie(
            cat,
            names="category",
            values="sales",
            hole=.5,
            title="Category Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sub = (
            filtered.groupby("sub_category")["sales"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        fig4 = px.bar(
            sub,
            x="sales",
            y="sub_category",
            orientation="h",
            title="Top 10 Sub Categories"
        )

        st.plotly_chart(fig4, use_container_width=True)

with tab2:

    st.subheader("Prophet Forecast")

    hist = forecast[forecast["type"] == "historical"]
    test = forecast[forecast["type"] == "test"]
    future = forecast[forecast["type"] == "future"]

    fig = go.Figure()

    if len(hist):
        fig.add_trace(
            go.Scatter(
                x=hist["ds"],
                y=hist["actual"],
                name="Historical"
            )
        )

    if len(test):
        fig.add_trace(
            go.Scatter(
                x=test["ds"],
                y=test["predicted"],
                name="Test Forecast"
            )
        )

    if len(future):
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["predicted"],
                name="Future Forecast"
            )
        )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    if len(future):
        growth = future["predicted"].iloc[-1] - future["predicted"].iloc[0]

        st.success(
            f"Forecasted Growth: ${growth:,.0f}"
        )

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        seg = (
            filtered.groupby("segment")["sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            seg,
            names="segment",
            values="sales",
            hole=.5,
            title="Customer Segments"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        ship = (
            filtered.groupby("ship_mode")["sales"]
            .sum()
            .reset_index()
        )

        fig2 = px.pie(
            ship,
            names="ship_mode",
            values="sales",
            hole=.5,
            title="Ship Mode Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Discount vs Profit")

    sample = filtered.sample(min(2000, len(filtered)), random_state=42)

    fig3 = px.scatter(
        sample,
        x="discount",
        y="profit",
        color="category",
        size="sales"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top Products")

    top_products = (
        filtered.groupby("product_name")["sales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    st.dataframe(top_products, use_container_width=True)

    best_region = (
        filtered.groupby("region")["sales"]
        .sum()
        .idxmax()
    )

    best_category = (
        filtered.groupby("category")["sales"]
        .sum()
        .idxmax()
    )

    st.info(
        f"""
🤖 AI Business Insights

• Best Region: {best_region}

• Best Category: {best_category}

• Revenue: ${revenue:,.0f}

• Profit: ${profit:,.0f}

• Profit Margin: {margin:.2f}%
"""
    )

with tab4:

    st.subheader("Data Explorer")

    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False)

    st.download_button(
        "Download Filtered Data",
        csv,
        "filtered_sales.csv",
        "text/csv"
    )

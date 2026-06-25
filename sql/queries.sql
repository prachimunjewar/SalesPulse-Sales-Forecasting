-- ============================================================
-- SalesPulse SQL Analytics Queries
-- Run these after loading superstore.csv into SQLite
-- ============================================================

-- 1. Total Revenue by Year and Category
SELECT
    strftime('%Y', order_date) AS year,
    category,
    ROUND(SUM(sales), 2)       AS total_sales,
    ROUND(SUM(profit), 2)      AS total_profit,
    COUNT(DISTINCT order_id)   AS total_orders
FROM superstore
GROUP BY year, category
ORDER BY year, total_sales DESC;


-- 2. Monthly Sales Trend (for time series input)
SELECT
    strftime('%Y-%m', order_date) AS month,
    ROUND(SUM(sales), 2)          AS monthly_sales,
    COUNT(DISTINCT order_id)      AS order_count,
    ROUND(AVG(sales), 2)          AS avg_order_value
FROM superstore
GROUP BY month
ORDER BY month;


-- 3. Top 10 Sub-Categories by Revenue
SELECT
    sub_category,
    ROUND(SUM(sales), 2)   AS total_sales,
    ROUND(SUM(profit), 2)  AS total_profit,
    ROUND(SUM(profit) * 100.0 / SUM(sales), 2) AS profit_margin_pct
FROM superstore
GROUP BY sub_category
ORDER BY total_sales DESC
LIMIT 10;


-- 4. Regional Performance Summary
SELECT
    region,
    segment,
    ROUND(SUM(sales), 2)         AS total_sales,
    ROUND(SUM(profit), 2)        AS total_profit,
    COUNT(DISTINCT customer_id)  AS unique_customers,
    COUNT(DISTINCT order_id)     AS total_orders
FROM superstore
GROUP BY region, segment
ORDER BY region, total_sales DESC;


-- 5. Seasonality — Sales by Quarter
SELECT
    strftime('%Y', order_date)                 AS year,
    CASE
        WHEN CAST(strftime('%m', order_date) AS INT) BETWEEN 1 AND 3  THEN 'Q1'
        WHEN CAST(strftime('%m', order_date) AS INT) BETWEEN 4 AND 6  THEN 'Q2'
        WHEN CAST(strftime('%m', order_date) AS INT) BETWEEN 7 AND 9  THEN 'Q3'
        ELSE 'Q4'
    END                                        AS quarter,
    ROUND(SUM(sales), 2)                       AS quarterly_sales
FROM superstore
GROUP BY year, quarter
ORDER BY year, quarter;


-- 6. Discount Impact on Profit
SELECT
    CASE
        WHEN discount = 0          THEN 'No Discount'
        WHEN discount <= 0.1       THEN '1-10%'
        WHEN discount <= 0.2       THEN '11-20%'
        WHEN discount <= 0.3       THEN '21-30%'
        ELSE 'Above 30%'
    END                            AS discount_band,
    COUNT(*)                       AS order_count,
    ROUND(SUM(sales), 2)           AS total_sales,
    ROUND(SUM(profit), 2)          AS total_profit,
    ROUND(AVG(profit_margin), 2)   AS avg_margin
FROM (
    SELECT *,
           ROUND(profit * 100.0 / NULLIF(sales, 0), 2) AS profit_margin
    FROM superstore
)
GROUP BY discount_band
ORDER BY total_profit DESC;


-- 7. YoY Revenue Growth by Category
WITH yearly AS (
    SELECT
        strftime('%Y', order_date) AS year,
        category,
        SUM(sales)                 AS total_sales
    FROM superstore
    GROUP BY year, category
)
SELECT
    a.year,
    a.category,
    ROUND(a.total_sales, 2)                                                           AS current_sales,
    ROUND(b.total_sales, 2)                                                           AS prev_sales,
    ROUND((a.total_sales - b.total_sales) * 100.0 / NULLIF(b.total_sales, 0), 2)     AS yoy_growth_pct
FROM yearly a
LEFT JOIN yearly b
    ON a.category = b.category
    AND CAST(a.year AS INT) = CAST(b.year AS INT) + 1
ORDER BY a.year, a.category;


-- 8. Shipping Mode Performance
SELECT
    ship_mode,
    COUNT(*)                 AS total_orders,
    ROUND(SUM(sales), 2)     AS total_sales,
    ROUND(AVG(
        julianday(ship_date) - julianday(order_date)
    ), 1)                    AS avg_ship_days
FROM superstore
GROUP BY ship_mode
ORDER BY total_sales DESC;

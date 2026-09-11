-- ============================================================================
-- RetailIQ - Core Business KPIs
-- ============================================================================

SELECT
    COUNT(DISTINCT o.order_id) AS total_orders,

    COUNT(DISTINCT CASE
        WHEN oi.order_id IS NOT NULL THEN o.order_id
    END) AS revenue_orders,

    COUNT(DISTINCT c.customer_unique_id) AS unique_customers,

    ROUND(
        SUM(oi.price + oi.freight_value),
        2
    ) AS total_revenue,

    ROUND(
        SUM(oi.price + oi.freight_value)
        / COUNT(DISTINCT oi.order_id),
        2
    ) AS average_order_value,

    COUNT(oi.order_id) AS total_items_sold

FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id

LEFT JOIN order_items oi
    ON o.order_id = oi.order_id;
-- ============================================================================
-- Monthly Revenue Trend
-- ============================================================================

SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::DATE AS month,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT oi.order_id) AS revenue_orders,
    COUNT(oi.order_id) AS items_sold,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY DATE_TRUNC('month', o.order_purchase_timestamp)
ORDER BY month;
-- ============================================================================
-- Revenue by Product Category
-- ============================================================================

SELECT
    COALESCE(
        c.category_name_english,
        p.product_category_name,
        'Unknown'
    ) AS category,
    COUNT(DISTINCT o.order_id) AS orders,
    COUNT(oi.order_id) AS items_sold,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
    ROUND(
        100.0 * SUM(oi.price + oi.freight_value)
        / SUM(SUM(oi.price + oi.freight_value)) OVER (),
        2
    ) AS revenue_share_pct
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
JOIN products p
    ON oi.product_id = p.product_id
LEFT JOIN categories c
    ON p.product_category_name = c.category_name_portuguese
WHERE o.order_status = 'delivered'
GROUP BY
    COALESCE(
        c.category_name_english,
        p.product_category_name,
        'Unknown'
    )
ORDER BY revenue DESC;
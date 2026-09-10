import pandas as pd

ORDERS_PATH = r"F:\RetailIQ\data\raw\olist\olist_orders_dataset.csv"

print("=" * 80)
print("RETAILIQ - ORDER DATE ANOMALY INVESTIGATION")
print("=" * 80)

orders = pd.read_csv(ORDERS_PATH)

date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# -------------------------------------------------------------------
# 1. Carrier date before purchase
# -------------------------------------------------------------------

anomalies = orders[
    orders["order_delivered_carrier_date"]
    < orders["order_purchase_timestamp"]
].copy()

print("\n" + "-" * 80)
print("CARRIER BEFORE PURCHASE")
print("-" * 80)

print(f"Anomalous orders: {len(anomalies):,}")

print("\nOrder status:")
print(anomalies["order_status"].value_counts())

print("\nSample anomalies:")
print(
    anomalies[
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ]
    .sort_values("order_purchase_timestamp")
    .head(20)
    .to_string(index=False)
)

# -------------------------------------------------------------------
# 2. Calculate how large the anomaly is
# -------------------------------------------------------------------

anomalies["carrier_lead_hours"] = (
    anomalies["order_purchase_timestamp"]
    - anomalies["order_delivered_carrier_date"]
).dt.total_seconds() / 3600

print("\n" + "-" * 80)
print("ANOMALY MAGNITUDE")
print("-" * 80)

print(
    anomalies["carrier_lead_hours"]
    .describe()
)

# -------------------------------------------------------------------
# 3. Check whether anomalous orders are delivered
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("DELIVERY STATUS OF ANOMALOUS ORDERS")
print("-" * 80)

print(
    anomalies["order_delivered_customer_date"]
    .notna()
    .value_counts()
    .rename({
        True: "Has customer delivery date",
        False: "Missing customer delivery date"
    })
)

# -------------------------------------------------------------------
# 4. Save anomalies for investigation
# -------------------------------------------------------------------

OUTPUT_PATH = r"F:\RetailIQ\data\processed\order_date_anomalies.csv"

anomalies.to_csv(OUTPUT_PATH, index=False)

print("\n" + "-" * 80)
print("SAVED")
print("-" * 80)

print(f"File: {OUTPUT_PATH}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
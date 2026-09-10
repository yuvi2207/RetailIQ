from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/olist")

orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")

print("=" * 80)
print("RETAILIQ - ORDER QUALITY VALIDATION")
print("=" * 80)

# ------------------------------------------------------------
# ORDER STATUS
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("ORDER STATUS")
print("-" * 80)

print(
    orders["order_status"]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------
# DATE CONVERSION
# ------------------------------------------------------------

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

print("\n" + "-" * 80)
print("DATE RANGE")
print("-" * 80)

for col in date_columns:
    print(
        f"\n{col}:"
        f"\n  Min: {orders[col].min()}"
        f"\n  Max: {orders[col].max()}"
        f"\n  Missing: {orders[col].isna().sum():,}"
    )

# ------------------------------------------------------------
# LOGICAL DATE CHECKS
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("DATE LOGIC CHECKS")
print("-" * 80)

checks = {
    "approved_before_purchase":
        orders["order_approved_at"] < orders["order_purchase_timestamp"],

    "carrier_before_purchase":
        orders["order_delivered_carrier_date"] < orders["order_purchase_timestamp"],

    "delivered_before_purchase":
        orders["order_delivered_customer_date"] < orders["order_purchase_timestamp"],

    "estimated_before_purchase":
        orders["order_estimated_delivery_date"] < orders["order_purchase_timestamp"],
}

for name, condition in checks.items():
    count = condition.fillna(False).sum()
    print(f"{name}: {count:,}")

# ------------------------------------------------------------
# DELIVERY PERFORMANCE
# ------------------------------------------------------------

delivered = orders[
    orders["order_delivered_customer_date"].notna()
].copy()

delivered["delivery_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

delivered["estimated_delay_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

print("\n" + "-" * 80)
print("DELIVERY METRICS")
print("-" * 80)

print(f"\nDelivered orders: {len(delivered):,}")

print(
    f"Average delivery time: "
    f"{delivered['delivery_days'].mean():.2f} days"
)

print(
    f"Median delivery time: "
    f"{delivered['delivery_days'].median():.2f} days"
)

print(
    f"Delivered after estimated date: "
    f"{(delivered['estimated_delay_days'] > 0).sum():,}"
)

print(
    f"Delivered before/on estimated date: "
    f"{(delivered['estimated_delay_days'] <= 0).sum():,}"
)

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
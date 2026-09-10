from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/olist")

customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")


print("=" * 80)
print("RETAILIQ - KEY & CUSTOMER IDENTITY VALIDATION")
print("=" * 80)


# -------------------------------------------------------------------
# COMPOSITE KEY CHECKS
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("COMPOSITE KEY CHECKS")
print("-" * 80)


# Order items
item_duplicates = items.duplicated(
    subset=["order_id", "order_item_id"]
).sum()

print("\norder_items")
print("  Key: (order_id, order_item_id)")
print(f"  Duplicate composite keys: {item_duplicates:,}")


# Payments
payment_duplicates = payments.duplicated(
    subset=["order_id", "payment_sequential"]
).sum()

print("\npayments")
print("  Key: (order_id, payment_sequential)")
print(f"  Duplicate composite keys: {payment_duplicates:,}")


# -------------------------------------------------------------------
# CUSTOMER IDENTITY
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("CUSTOMER IDENTITY")
print("-" * 80)

unique_customers = customers["customer_unique_id"].nunique()
customer_ids = customers["customer_id"].nunique()

print(f"\ncustomer_id unique values:        {customer_ids:,}")
print(f"customer_unique_id unique values: {unique_customers:,}")

print(
    f"\nAverage customer records per unique customer: "
    f"{len(customers) / unique_customers:.2f}"
)


# -------------------------------------------------------------------
# ORDERS PER REAL CUSTOMER
# -------------------------------------------------------------------

orders_with_customer = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

orders_per_real_customer = (
    orders_with_customer
    .groupby("customer_unique_id")
    .size()
)

print("\nOrders per customer_unique_id:")
print(f"  Unique customers with orders: {orders_per_real_customer.size:,}")
print(
    f"  Maximum orders by one real customer: "
    f"{orders_per_real_customer.max()}"
)

print(
    f"  Customers with >1 order: "
    f"{(orders_per_real_customer > 1).sum():,}"
)


# -------------------------------------------------------------------
# REVIEW DUPLICATES
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("REVIEW IDENTIFIERS")
print("-" * 80)

print(
    f"\nreview_id unique values: "
    f"{reviews['review_id'].nunique():,}"
)

print(
    f"review_id duplicate rows: "
    f"{reviews['review_id'].duplicated().sum():,}"
)


print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
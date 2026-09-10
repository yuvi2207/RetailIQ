from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/olist")

customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

print("=" * 80)
print("RETAILIQ - RELATIONSHIP VALIDATION")
print("=" * 80)


def check_unique(df, column, table):
    duplicates = df[column].duplicated().sum()

    print(f"\n{table}.{column}")
    print(f"  Unique values: {df[column].nunique():,}")
    print(f"  Duplicate rows: {duplicates:,}")

    if duplicates == 0:
        print("  ✓ Unique")
    else:
        print("  ⚠ Not unique")


def check_foreign_key(child_df, child_column, parent_df, parent_column, relationship):
    child_values = set(child_df[child_column].dropna().unique())
    parent_values = set(parent_df[parent_column].dropna().unique())

    missing = child_values - parent_values

    print(f"\n{relationship}")
    print(f"  Child unique values: {len(child_values):,}")
    print(f"  Parent unique values: {len(parent_values):,}")
    print(f"  Unmatched values: {len(missing):,}")

    if len(missing) == 0:
        print("  ✓ Relationship valid")
    else:
        print("  ⚠ Unmatched values found")


# -------------------------------------------------------------------
# PRIMARY KEY CHECKS
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("PRIMARY KEY CHECKS")
print("-" * 80)

check_unique(customers, "customer_id", "customers")
check_unique(orders, "order_id", "orders")
check_unique(products, "product_id", "products")
check_unique(sellers, "seller_id", "sellers")


# -------------------------------------------------------------------
# FOREIGN KEY CHECKS
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("FOREIGN KEY CHECKS")
print("-" * 80)

check_foreign_key(
    orders,
    "customer_id",
    customers,
    "customer_id",
    "orders.customer_id → customers.customer_id"
)

check_foreign_key(
    items,
    "order_id",
    orders,
    "order_id",
    "order_items.order_id → orders.order_id"
)

check_foreign_key(
    items,
    "product_id",
    products,
    "product_id",
    "order_items.product_id → products.product_id"
)

check_foreign_key(
    items,
    "seller_id",
    sellers,
    "seller_id",
    "order_items.seller_id → sellers.seller_id"
)

check_foreign_key(
    payments,
    "order_id",
    orders,
    "order_id",
    "payments.order_id → orders.order_id"
)

check_foreign_key(
    reviews,
    "order_id",
    orders,
    "order_id",
    "reviews.order_id → orders.order_id"
)


# -------------------------------------------------------------------
# CARDINALITY
# -------------------------------------------------------------------

print("\n" + "-" * 80)
print("CARDINALITY CHECKS")
print("-" * 80)

print("\nOrders per customer:")
orders_per_customer = orders.groupby("customer_id").size()

print(f"  Customers with orders: {orders_per_customer.size:,}")
print(f"  Maximum orders by one customer_id: {orders_per_customer.max()}")

print("\nItems per order:")
items_per_order = items.groupby("order_id").size()

print(f"  Orders containing items: {items_per_order.size:,}")
print(f"  Maximum items in one order: {items_per_order.max()}")

print("\nPayments per order:")
payments_per_order = payments.groupby("order_id").size()

print(f"  Orders with payment records: {payments_per_order.size:,}")
print(f"  Maximum payment records in one order: {payments_per_order.max()}")

print("\nReviews per order:")
reviews_per_order = reviews.groupby("order_id").size()

print(f"  Orders with reviews: {reviews_per_order.size:,}")
print(f"  Maximum reviews for one order: {reviews_per_order.max()}")


print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
import os
import pandas as pd
import psycopg2
from io import StringIO
from dotenv import load_dotenv

# ============================================================================
# RetailIQ - PostgreSQL ETL Loader
# ============================================================================

load_dotenv()

BASE_DIR = r"F:\RetailIQ"
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "olist")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_csv_to_postgres(connection, csv_file, table_name, columns):
    path = os.path.join(DATA_DIR, csv_file)

    print(f"\nLoading {csv_file} → {table_name}")

    df = pd.read_csv(path)

    # Map source CSV column names to cleaned PostgreSQL column names
    column_mapping = {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length"
    }

    # Category translation CSV needs a different mapping
    if table_name == "categories":
        column_mapping = {
            "product_category_name": "category_name_portuguese",
            "product_category_name_english": "category_name_english"
        }

    df = df.rename(columns=column_mapping)

    # Keep only the columns expected by PostgreSQL
    df = df[columns]

    # Convert product integer columns to proper integers
    if table_name == "products":
        integer_columns = [
            "product_name_length",
            "product_description_length",
            "product_photos_qty"
        ]

        for column in integer_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).round().astype("Int64")

    # Convert pandas missing values to PostgreSQL NULL
    df = df.where(pd.notnull(df), None)

    buffer = StringIO()

    df.to_csv(
        buffer,
        index=False,
        header=False,
        na_rep="\\N"
    )

    buffer.seek(0)

    cursor = connection.cursor()

    copy_sql = f"""
        COPY {table_name} ({', '.join(columns)})
        FROM STDIN
        WITH (
            FORMAT CSV,
            NULL '\\N'
        )
    """

    cursor.copy_expert(copy_sql, buffer)

    connection.commit()

    cursor.close()

    print(f"Loaded {len(df):,} rows")


def main():
    print("=" * 80)
    print("RETAILIQ - POSTGRESQL ETL LOADER")
    print("=" * 80)

    connection = get_connection()

    try:
        load_csv_to_postgres(
            connection,
            "olist_customers_dataset.csv",
            "customers",
            [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_products_dataset.csv",
            "products",
            [
                "product_id",
                "product_category_name",
                "product_name_length",
                "product_description_length",
                "product_photos_qty",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_sellers_dataset.csv",
            "sellers",
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state"
            ]
        )

        load_csv_to_postgres(
            connection,
            "product_category_name_translation.csv",
            "categories",
            [
                "category_name_portuguese",
                "category_name_english"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_orders_dataset.csv",
            "orders",
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_order_items_dataset.csv",
            "order_items",
            [
                "order_id",
                "order_item_id",
                "product_id",
                "seller_id",
                "shipping_limit_date",
                "price",
                "freight_value"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_order_payments_dataset.csv",
            "payments",
            [
                "order_id",
                "payment_sequential",
                "payment_type",
                "payment_installments",
                "payment_value"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_order_reviews_dataset.csv",
            "reviews",
            [
                "review_id",
                "order_id",
                "review_score",
                "review_comment_title",
                "review_comment_message",
                "review_creation_date",
                "review_answer_timestamp"
            ]
        )

        load_csv_to_postgres(
            connection,
            "olist_geolocation_dataset.csv",
            "geolocation",
            [
                "geolocation_zip_code_prefix",
                "geolocation_lat",
                "geolocation_lng",
                "geolocation_city",
                "geolocation_state"
            ]
        )

        print("\n" + "=" * 80)
        print("ETL COMPLETE")
        print("=" * 80)

    except Exception as e:
        connection.rollback()

        print("\n" + "=" * 80)
        print("ETL FAILED")
        print("=" * 80)
        print(e)

        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()
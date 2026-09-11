import os
import joblib
import numpy as np
import pandas as pd
import psycopg2

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================================
# RetailIQ - RFM Customer Segmentation
# Step 1: Load, transform, scale, evaluate, and segment customers
# ============================================================================


load_dotenv()


# ============================================================================
# Database configuration
# ============================================================================

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}


# ============================================================================
# Database connection
# ============================================================================

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ============================================================================
# Load RFM data
# ============================================================================

def load_rfm_data():

    connection = get_connection()

    query = """
        SELECT
            customer_unique_id,
            recency,
            frequency,
            monetary
        FROM customer_rfm
    """

    df = pd.read_sql_query(query, connection)

    connection.close()

    return df


# ============================================================================
# Main pipeline
# ============================================================================

def main():

    print("=" * 80)
    print("RETAILIQ - RFM CUSTOMER SEGMENTATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Load RFM data
    # ------------------------------------------------------------------------

    df = load_rfm_data()

    print(f"\nCustomers loaded: {len(df):,}")

    print("\nFirst 5 rows:")
    print(df.head())

    # ------------------------------------------------------------------------
    # 2. Data types
    # ------------------------------------------------------------------------

    print("\nData types:")
    print(df.dtypes)

    # ------------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------------

    print("\nMissing values:")
    print(df.isnull().sum())

    # ------------------------------------------------------------------------
    # 4. RFM summary
    # ------------------------------------------------------------------------

    print("\nRFM summary:")

    print(
        df[
            ["recency", "frequency", "monetary"]
        ].describe()
    )

    # ------------------------------------------------------------------------
    # 5. Original skewness
    # ------------------------------------------------------------------------

    print("\nRFM skewness:")

    print(
        df[
            ["recency", "frequency", "monetary"]
        ].skew()
    )

    # ------------------------------------------------------------------------
    # 6. Cap extreme frequency values
    # ------------------------------------------------------------------------

    frequency_cap = df["frequency"].quantile(0.99)

    df["frequency_capped"] = df["frequency"].clip(
        upper=frequency_cap
    )

    print(
        f"\nFrequency 99th percentile cap: "
        f"{frequency_cap:.2f}"
    )

    # ------------------------------------------------------------------------
    # 7. Log-transform skewed features
    # ------------------------------------------------------------------------

    df["frequency_log"] = np.log1p(
        df["frequency_capped"]
    )

    df["monetary_log"] = np.log1p(
        df["monetary"]
    )

    print("\nTransformed RFM skewness:")

    print(
        df[
            [
                "recency",
                "frequency_log",
                "monetary_log"
            ]
        ].skew()
    )

    # ------------------------------------------------------------------------
    # 8. Select features for clustering
    # ------------------------------------------------------------------------

    features = df[
        [
            "recency",
            "frequency_log",
            "monetary_log"
        ]
    ]

    # ------------------------------------------------------------------------
    # 9. Standardize features
    # ------------------------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[
            "recency_scaled",
            "frequency_scaled",
            "monetary_scaled"
        ]
    )

    print("\nScaled feature statistics:")

    print(
        scaled_df.describe()
    )

    # ------------------------------------------------------------------------
    # 10. Test K values using inertia
    # ------------------------------------------------------------------------

    print("\nTesting K values...")

    inertia_values = []

    for k in range(2, 9):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X_scaled)

        inertia_values.append(
            kmeans.inertia_
        )

        print(
            f"K={k} | "
            f"Inertia={kmeans.inertia_:,.2f}"
        )

    # ------------------------------------------------------------------------
    # 11. Test K values using Silhouette Score
    # ------------------------------------------------------------------------

    print("\nTesting Silhouette Scores...")

    silhouette_values = []

    for k in range(2, 9):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        silhouette_values.append(score)

        print(
            f"K={k} | "
            f"Silhouette Score={score:.4f}"
        )

    # ------------------------------------------------------------------------
    # 12. Display best silhouette score
    # ------------------------------------------------------------------------

    best_k = range(2, 9)[
        np.argmax(silhouette_values)
    ]

    best_score = max(silhouette_values)

    print(
        f"\nBest silhouette score: "
        f"K={best_k} | Score={best_score:.4f}"
    )

    # ------------------------------------------------------------------------
    # 13. Final customer segmentation
    # ------------------------------------------------------------------------
    #
    # We intentionally use K=4 rather than automatically selecting K=2.
    #
    # K=2 has the highest silhouette score, but two clusters are too coarse
    # for useful customer segmentation. K=4 provides more actionable
    # behavioral groups.
    # ------------------------------------------------------------------------

    final_k = 4

    print(
        f"\nCreating final K={final_k} "
        "customer segments..."
    )

    final_kmeans = KMeans(
        n_clusters=final_k,
        random_state=42,
        n_init=10
    )

    df["cluster"] = final_kmeans.fit_predict(
        X_scaled
    )

    # ------------------------------------------------------------------------
    # 14. Cluster distribution
    # ------------------------------------------------------------------------

    print("\nCluster distribution:")

    cluster_counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    print(cluster_counts)

    # ------------------------------------------------------------------------
    # 15. Cluster RFM profile
    # ------------------------------------------------------------------------

    cluster_profile = (
        df.groupby("cluster")
        .agg(
            customers=(
                "customer_unique_id",
                "count"
            ),

            avg_recency=(
                "recency",
                "mean"
            ),

            median_recency=(
                "recency",
                "median"
            ),

            avg_frequency=(
                "frequency",
                "mean"
            ),

            median_frequency=(
                "frequency",
                "median"
            ),

            avg_monetary=(
                "monetary",
                "mean"
            ),

            median_monetary=(
                "monetary",
                "median"
            )
        )
        .round(2)
    )

    # ------------------------------------------------------------------------
    # 16. Customer percentage
    # ------------------------------------------------------------------------

    cluster_profile["customer_percentage"] = (
        cluster_profile["customers"]
        / len(df)
        * 100
    ).round(2)

    print("\nCluster RFM Profile:")

    print(cluster_profile)

    # ------------------------------------------------------------------------
    # 17. Cluster centers
    # ------------------------------------------------------------------------

    cluster_centers = pd.DataFrame(
        final_kmeans.cluster_centers_,
        columns=[
            "recency_scaled",
            "frequency_scaled",
            "monetary_scaled"
        ]
    )

    print("\nCluster Centers:")

    print(
        cluster_centers.round(3)
    )

    # ------------------------------------------------------------------------
    # 18. Assign business-friendly segment names
    # ------------------------------------------------------------------------
    #
    # Based on the observed cluster profiles:
    #
    # Cluster 2 -> high frequency + high monetary
    # Cluster 3 -> recent + low monetary
    # Cluster 0 -> moderate recency + high monetary
    # Cluster 1 -> high recency + lower monetary
    #
    # These names are based on behavioral characteristics, not arbitrary
    # cluster numbers.
    # ------------------------------------------------------------------------

    segment_names = {
        0: "High Value",
        1: "At Risk",
        2: "Loyal High Value",
        3: "Recent Low Value"
    }

    df["segment"] = df["cluster"].map(
        segment_names
    )

    cluster_profile["segment"] = (
        cluster_profile.index.map(
            segment_names
        )
    )

    # Reorder columns

    cluster_profile = cluster_profile[
        [
            "segment",
            "customers",
            "customer_percentage",
            "avg_recency",
            "median_recency",
            "avg_frequency",
            "median_frequency",
            "avg_monetary",
            "median_monetary"
        ]
    ]

    # ------------------------------------------------------------------------
    # 19. Final segment summary
    # ------------------------------------------------------------------------

    print("\nFinal Customer Segments:")

    print(
        cluster_profile
        .sort_values(
            "avg_monetary",
            ascending=False
        )
    )

    # ------------------------------------------------------------------------
    # 20. Save segmented customer data
    # ------------------------------------------------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_customer_file = (
        "outputs/customer_segments.csv"
    )

    df.to_csv(
        output_customer_file,
        index=False
    )

    print(
        f"\nSaved customer segments to: "
        f"{output_customer_file}"
    )

    # ------------------------------------------------------------------------
    # 21. Save cluster profile
    # ------------------------------------------------------------------------

    output_profile_file = (
        "outputs/cluster_profile.csv"
    )

    cluster_profile.to_csv(
        output_profile_file
    )

    print(
        f"Saved cluster profile to: "
        f"{output_profile_file}"
    )

    # ------------------------------------------------------------------------
    # 22. Save model
    # ------------------------------------------------------------------------

    joblib.dump(
        final_kmeans,
        "outputs/kmeans_model.pkl"
    )

    print(
        "Saved K-Means model to: "
        "outputs/kmeans_model.pkl"
    )

    # ------------------------------------------------------------------------
    # 23. Save scaler
    # ------------------------------------------------------------------------

    joblib.dump(
        scaler,
        "outputs/rfm_scaler.pkl"
    )

    print(
        "Saved scaler to: "
        "outputs/rfm_scaler.pkl"
    )

    # ------------------------------------------------------------------------
    # 24. Final summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RFM CUSTOMER SEGMENTATION COMPLETE")
    print("=" * 80)

    print(
        f"\nCustomers segmented: {len(df):,}"
    )

    print(
        f"Final clusters: {final_k}"
    )

    print(
        f"Best silhouette K: {best_k}"
    )

    print(
        f"Best silhouette score: {best_score:.4f}"
    )

    print("\nOutput files:")

    print(
        "1. outputs/customer_segments.csv"
    )

    print(
        "2. outputs/cluster_profile.csv"
    )

    print(
        "3. outputs/kmeans_model.pkl"
    )

    print(
        "4. outputs/rfm_scaler.pkl"
    )


if __name__ == "__main__":
    main()
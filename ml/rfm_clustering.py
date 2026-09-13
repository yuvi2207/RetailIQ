import os
import pickle
import numpy as np
import pandas as pd
import psycopg2

from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


# ============================================================================
# RetailIQ - RFM Customer Segmentation
#
# Complete ML pipeline:
# 1. Load RFM data from PostgreSQL
# 2. Validate input data
# 3. Handle skewed RFM variables
# 4. Scale features
# 5. Evaluate K-Means for K=2..8
# 6. Calculate:
#       - Inertia
#       - Silhouette Score
#       - Calinski-Harabasz Score
#       - Davies-Bouldin Score
# 7. Validate K=4 stability using multiple random seeds
# 8. Create final K=4 business segments
# 9. Assign robust business labels based on actual cluster profiles
# 10. Save customer segments and ML artifacts
#
# IMPORTANT:
# K-Means cluster IDs are arbitrary.
# Business labels are therefore assigned from cluster characteristics,
# NOT from hard-coded cluster numbers.
# ============================================================================


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


OUTPUT_DIR = "outputs"

FINAL_K = 4

K_RANGE = range(2, 9)

RANDOM_STATE = 42

STABILITY_SEEDS = [42, 7, 21, 100, 123]


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

    try:

        df = pd.read_sql_query(
            query,
            connection
        )

    finally:

        connection.close()

    return df


# ============================================================================
# Input validation
# ============================================================================

def validate_input_data(df):

    print()
    print("=" * 80)
    print("INPUT DATA VALIDATION")
    print("=" * 80)

    print(f"\nCustomers loaded: {len(df):,}")

    # ------------------------------------------------------------------------
    # Duplicate customers
    # ------------------------------------------------------------------------

    duplicate_customers = (
        df["customer_unique_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate customer IDs: "
        f"{duplicate_customers:,}"
    )

    if duplicate_customers > 0:

        raise ValueError(
            "Duplicate customer IDs detected. "
            "RFM table should contain one row per customer."
        )

    # ------------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------------

    print("\nMissing values:")

    print(
        df[
            [
                "customer_unique_id",
                "recency",
                "frequency",
                "monetary",
            ]
        ]
        .isnull()
        .sum()
    )

    if df[
        [
            "customer_unique_id",
            "recency",
            "frequency",
            "monetary",
        ]
    ].isnull().any().any():

        raise ValueError(
            "Missing values detected in RFM data."
        )

    # ------------------------------------------------------------------------
    # Invalid values
    # ------------------------------------------------------------------------

    negative_recency = (
        df["recency"] < 0
    ).sum()

    invalid_frequency = (
        df["frequency"] <= 0
    ).sum()

    invalid_monetary = (
        df["monetary"] <= 0
    ).sum()

    print("\nInvalid values:")

    print(
        f"Negative recency: "
        f"{negative_recency:,}"
    )

    print(
        f"Invalid frequency: "
        f"{invalid_frequency:,}"
    )

    print(
        f"Invalid monetary: "
        f"{invalid_monetary:,}"
    )

    if negative_recency > 0:

        raise ValueError(
            "Negative recency values detected."
        )

    if invalid_frequency > 0:

        raise ValueError(
            "Frequency must be greater than zero."
        )

    if invalid_monetary > 0:

        raise ValueError(
            "Monetary values must be greater than zero."
        )

    # ------------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------------

    print("\nData types:")

    print(df.dtypes)

    print("\nInput validation passed.")


# ============================================================================
# RFM preprocessing
# ============================================================================

def preprocess_rfm(df):

    print()
    print("=" * 80)
    print("RFM PREPROCESSING")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print("\nRFM summary:")

    print(
        df[
            [
                "recency",
                "frequency",
                "monetary",
            ]
        ]
        .describe()
    )

    # ------------------------------------------------------------------------
    # Original skewness
    # ------------------------------------------------------------------------

    print("\nOriginal RFM skewness:")

    print(
        df[
            [
                "recency",
                "frequency",
                "monetary",
            ]
        ]
        .skew()
    )

    # ------------------------------------------------------------------------
    # Cap extreme frequency values
    # ------------------------------------------------------------------------
    #
    # Frequency is extremely right-skewed.
    # The 99th percentile is used as an outlier cap.
    # ------------------------------------------------------------------------

    frequency_cap = (
        df["frequency"]
        .quantile(0.99)
    )

    df["frequency_capped"] = (
        df["frequency"]
        .clip(
            upper=frequency_cap
        )
    )

    affected_customers = (
        df["frequency"]
        > frequency_cap
    ).sum()

    print(
        f"\nFrequency 99th percentile cap: "
        f"{frequency_cap:.2f}"
    )

    print(
        f"Customers affected by frequency cap: "
        f"{affected_customers:,}"
    )

    # ------------------------------------------------------------------------
    # Log transformation
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
                "monetary_log",
            ]
        ]
        .skew()
    )

    # ------------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------------

    features = df[
        [
            "recency",
            "frequency_log",
            "monetary_log",
        ]
    ].copy()

    # ------------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        features
    )

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[
            "recency_scaled",
            "frequency_scaled",
            "monetary_scaled",
        ],
    )

    print("\nScaled feature statistics:")

    print(
        scaled_df.describe()
    )

    return df, X_scaled, scaler


# ============================================================================
# K-Means evaluation
# ============================================================================

def evaluate_kmeans(X_scaled):

    print()
    print("=" * 80)
    print("K-MEANS MODEL VALIDATION")
    print("=" * 80)

    print(
        "\nEvaluating K values from 2 to 8..."
    )

    results = []

    for k in K_RANGE:

        print(
            f"\nEvaluating K={k}..."
        )

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(
            X_scaled
        )

        inertia = model.inertia_

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        calinski = (
            calinski_harabasz_score(
                X_scaled,
                labels,
            )
        )

        davies = (
            davies_bouldin_score(
                X_scaled,
                labels,
            )
        )

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies,
            }
        )

        print(
            f"K={k} | "
            f"Inertia={inertia:,.2f} | "
            f"Silhouette={silhouette:.4f} | "
            f"Calinski-Harabasz={calinski:,.2f} | "
            f"Davies-Bouldin={davies:.4f}"
        )

    metrics_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------------------
    # Best metrics
    # ------------------------------------------------------------------------

    best_silhouette = (
        metrics_df.loc[
            metrics_df[
                "silhouette_score"
            ].idxmax()
        ]
    )

    best_calinski = (
        metrics_df.loc[
            metrics_df[
                "calinski_harabasz_score"
            ].idxmax()
        ]
    )

    best_davies = (
        metrics_df.loc[
            metrics_df[
                "davies_bouldin_score"
            ].idxmin()
        ]
    )

    print()
    print("-" * 80)
    print("BEST VALIDATION RESULTS")
    print("-" * 80)

    print(
        f"Best Silhouette: "
        f"K={int(best_silhouette['k'])} | "
        f"{best_silhouette['silhouette_score']:.4f}"
    )

    print(
        f"Best Calinski-Harabasz: "
        f"K={int(best_calinski['k'])} | "
        f"{best_calinski['calinski_harabasz_score']:,.2f}"
    )

    print(
        f"Best Davies-Bouldin: "
        f"K={int(best_davies['k'])} | "
        f"{best_davies['davies_bouldin_score']:.4f}"
    )

    # ------------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "clustering_validation_metrics.csv",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"\nSaved validation metrics: "
        f"{metrics_path}"
    )

    return metrics_df


# ============================================================================
# K=4 stability validation
# ============================================================================

def validate_cluster_stability(
    X_scaled,
    reference_labels,
):

    print()
    print("=" * 80)
    print("K=4 CLUSTER STABILITY VALIDATION")
    print("=" * 80)

    stability_results = []

    for seed in STABILITY_SEEDS:

        model = KMeans(
            n_clusters=FINAL_K,
            random_state=seed,
            n_init=10,
        )

        labels = model.fit_predict(
            X_scaled
        )

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        ari = adjusted_rand_score(
            reference_labels,
            labels,
        )

        stability_results.append(
            {
                "seed": seed,
                "silhouette_score": silhouette,
                "adjusted_rand_index": ari,
            }
        )

        print(
            f"Seed={seed} | "
            f"Silhouette={silhouette:.4f} | "
            f"ARI={ari:.4f}"
        )

    stability_df = pd.DataFrame(
        stability_results
    )

    average_ari = (
        stability_df[
            "adjusted_rand_index"
        ]
        .mean()
    )

    average_silhouette = (
        stability_df[
            "silhouette_score"
        ]
        .mean()
    )

    print()
    print("-" * 80)

    print(
        f"Average K=4 ARI: "
        f"{average_ari:.4f}"
    )

    print(
        f"Average K=4 silhouette: "
        f"{average_silhouette:.4f}"
    )

    # ------------------------------------------------------------------------
    # Stability interpretation
    # ------------------------------------------------------------------------

    if average_ari >= 0.90:

        stability_level = "Excellent"

    elif average_ari >= 0.75:

        stability_level = "Good"

    elif average_ari >= 0.50:

        stability_level = "Moderate"

    else:

        stability_level = "Poor"

    print(
        f"Cluster stability: "
        f"{stability_level}"
    )

    stability_df[
        "stability_level"
    ] = stability_level

    stability_path = os.path.join(
        OUTPUT_DIR,
        "kmeans_stability.csv",
    )

    stability_df.to_csv(
        stability_path,
        index=False,
    )

    print(
        f"\nSaved stability results: "
        f"{stability_path}"
    )

    return (
        stability_df,
        average_ari,
        average_silhouette,
        stability_level,
    )


# ============================================================================
# Create final K=4 model
# ============================================================================

def create_final_clusters(
    df,
    X_scaled,
):

    print()
    print("=" * 80)
    print("FINAL BUSINESS SEGMENTATION")
    print("=" * 80)

    print(
        "\nCreating final K=4 customer segments..."
    )

    final_kmeans = KMeans(
        n_clusters=FINAL_K,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    df["cluster"] = (
        final_kmeans.fit_predict(
            X_scaled
        )
    )

    print("\nCluster distribution:")

    print(
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    # ------------------------------------------------------------------------
    # Cluster profile
    # ------------------------------------------------------------------------

    cluster_profile = (
        df.groupby("cluster")
        .agg(
            customers=(
                "customer_unique_id",
                "count",
            ),
            avg_recency=(
                "recency",
                "mean",
            ),
            median_recency=(
                "recency",
                "median",
            ),
            avg_frequency=(
                "frequency",
                "mean",
            ),
            median_frequency=(
                "frequency",
                "median",
            ),
            avg_monetary=(
                "monetary",
                "mean",
            ),
            median_monetary=(
                "monetary",
                "median",
            ),
        )
    )

    cluster_profile[
        "customer_percentage"
    ] = (
        cluster_profile[
            "customers"
        ]
        / len(df)
        * 100
    )

    # ------------------------------------------------------------------------
    # Round values
    # ------------------------------------------------------------------------

    cluster_profile = (
        cluster_profile[
            [
                "customers",
                "customer_percentage",
                "avg_recency",
                "median_recency",
                "avg_frequency",
                "median_frequency",
                "avg_monetary",
                "median_monetary",
            ]
        ]
    )

    print("\nCluster RFM Profile:")

    print(
        cluster_profile
        .round(2)
        .to_string()
    )

    return (
        df,
        cluster_profile,
        final_kmeans,
    )


# ============================================================================
# Robust business segment naming
# ============================================================================

def assign_business_segments(
    df,
    cluster_profile,
):

    print()
    print("=" * 80)
    print("BUSINESS SEGMENT LABELING")
    print("=" * 80)

    print(
        "\nAssigning labels from actual RFM cluster characteristics..."
    )

    # ------------------------------------------------------------------------
    # IMPORTANT
    #
    # K-Means cluster IDs are arbitrary.
    #
    # We therefore DO NOT write:
    #
    # cluster 0 = High Value
    # cluster 1 = At Risk
    #
    # Instead we inspect the actual RFM profile.
    # ------------------------------------------------------------------------

    segment_mapping = {}

    remaining_clusters = set(
        cluster_profile.index
    )

    # ------------------------------------------------------------------------
    # 1. Loyal High Value
    #
    # Highest purchase frequency.
    # ------------------------------------------------------------------------

    loyal_cluster = (
        cluster_profile[
            "avg_frequency"
        ]
        .idxmax()
    )

    segment_mapping[
        loyal_cluster
    ] = "Loyal High Value"

    remaining_clusters.remove(
        loyal_cluster
    )

    # ------------------------------------------------------------------------
    # 2. High Value
    #
    # Highest monetary value among the remaining clusters.
    # ------------------------------------------------------------------------

    high_value_cluster = (
        cluster_profile.loc[
            list(remaining_clusters),
            "avg_monetary",
        ]
        .idxmax()
    )

    segment_mapping[
        high_value_cluster
    ] = "High Value"

    remaining_clusters.remove(
        high_value_cluster
    )

    # ------------------------------------------------------------------------
    # 3. At Risk
    #
    # Highest recency = longest time since purchase.
    # ------------------------------------------------------------------------

    at_risk_cluster = (
        cluster_profile.loc[
            list(remaining_clusters),
            "avg_recency",
        ]
        .idxmax()
    )

    segment_mapping[
        at_risk_cluster
    ] = "At Risk"

    remaining_clusters.remove(
        at_risk_cluster
    )

    # ------------------------------------------------------------------------
    # 4. Recent Low Value
    #
    # The only remaining cluster.
    # ------------------------------------------------------------------------

    if len(remaining_clusters) != 1:

        raise ValueError(
            "Unable to identify a unique "
            "Recent Low Value cluster."
        )

    recent_low_value_cluster = (
        list(remaining_clusters)[0]
    )

    segment_mapping[
        recent_low_value_cluster
    ] = "Recent Low Value"

    # ------------------------------------------------------------------------
    # Apply labels
    # ------------------------------------------------------------------------

    df["segment"] = (
        df["cluster"]
        .map(segment_mapping)
    )

    cluster_profile[
        "segment"
    ] = (
        cluster_profile.index
        .map(segment_mapping)
    )

    # ------------------------------------------------------------------------
    # Validate mapping
    # ------------------------------------------------------------------------

    expected_segments = {
        "Loyal High Value",
        "High Value",
        "At Risk",
        "Recent Low Value",
    }

    actual_segments = set(
        cluster_profile[
            "segment"
        ]
    )

    if actual_segments != expected_segments:

        raise ValueError(
            "Business segment mapping failed.\n"
            f"Expected: {expected_segments}\n"
            f"Found: {actual_segments}"
        )

    if (
        cluster_profile[
            "segment"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Business segment mapping failed: "
            "duplicate segment labels detected."
        )

    if df["segment"].isnull().any():

        raise ValueError(
            "Some customers were not assigned "
            "a business segment."
        )

    print("\nValidated segment mapping:")

    print(
        cluster_profile[
            [
                "segment",
                "customers",
                "customer_percentage",
                "avg_recency",
                "avg_frequency",
                "avg_monetary",
            ]
        ]
        .sort_values("segment")
        .round(2)
        .to_string()
    )

    return (
        df,
        cluster_profile,
        segment_mapping,
    )


# ============================================================================
# Save final artifacts
# ============================================================================

def save_outputs(
    df,
    cluster_profile,
    final_kmeans,
    scaler,
    validation_metrics,
    stability_df,
):

    print()
    print("=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Customer segments
    # ------------------------------------------------------------------------

    customer_segments_path = os.path.join(
        OUTPUT_DIR,
        "customer_segments.csv",
    )

    df.to_csv(
        customer_segments_path,
        index=False,
    )

    print(
        f"Saved customer segments: "
        f"{customer_segments_path}"
    )

    # ------------------------------------------------------------------------
    # Cluster profile
    # ------------------------------------------------------------------------

    cluster_profile_path = os.path.join(
        OUTPUT_DIR,
        "cluster_profile.csv",
    )

    cluster_profile.round(
        2
    ).to_csv(
        cluster_profile_path
    )

    print(
        f"Saved cluster profile: "
        f"{cluster_profile_path}"
    )

    # ------------------------------------------------------------------------
    # K-Means model
    # ------------------------------------------------------------------------

    kmeans_path = os.path.join(
        OUTPUT_DIR,
        "kmeans_model.pkl",
    )

    with open(
        kmeans_path,
        "wb",
    ) as file:

        pickle.dump(
            final_kmeans,
            file,
        )

    print(
        f"Saved K-Means model: "
        f"{kmeans_path}"
    )

    # ------------------------------------------------------------------------
    # Scaler
    # ------------------------------------------------------------------------

    scaler_path = os.path.join(
        OUTPUT_DIR,
        "rfm_scaler.pkl",
    )

    with open(
        scaler_path,
        "wb",
    ) as file:

        pickle.dump(
            scaler,
            file,
        )

    print(
        f"Saved scaler: "
        f"{scaler_path}"
    )

    # ------------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------------

    validation_path = os.path.join(
        OUTPUT_DIR,
        "clustering_validation_summary.csv",
    )

    final_metrics = (
        validation_metrics[
            validation_metrics["k"]
            == FINAL_K
        ]
        .copy()
    )

    final_metrics.to_csv(
        validation_path,
        index=False,
    )

    print(
        f"Saved validation summary: "
        f"{validation_path}"
    )

    # ------------------------------------------------------------------------
    # Full validation metrics
    # ------------------------------------------------------------------------

    full_validation_path = os.path.join(
        OUTPUT_DIR,
        "clustering_validation_metrics.csv",
    )

    validation_metrics.to_csv(
        full_validation_path,
        index=False,
    )

    print(
        f"Saved full validation metrics: "
        f"{full_validation_path}"
    )

    # ------------------------------------------------------------------------
    # Stability results
    # ------------------------------------------------------------------------

    stability_path = os.path.join(
        OUTPUT_DIR,
        "kmeans_stability.csv",
    )

    stability_df.to_csv(
        stability_path,
        index=False,
    )

    print(
        f"Saved stability results: "
        f"{stability_path}"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 80)
    print("RETAILIQ - RFM CUSTOMER SEGMENTATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    print("\nLoading RFM data...")

    df = load_rfm_data()

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    validate_input_data(
        df
    )

    # ------------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------------

    (
        df,
        X_scaled,
        scaler,
    ) = preprocess_rfm(
        df
    )

    # ------------------------------------------------------------------------
    # Evaluate K values
    # ------------------------------------------------------------------------

    validation_metrics = (
        evaluate_kmeans(
            X_scaled
        )
    )

    # ------------------------------------------------------------------------
    # Temporary K=4 model for stability reference
    # ------------------------------------------------------------------------

    reference_model = KMeans(
        n_clusters=FINAL_K,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    reference_labels = (
        reference_model
        .fit_predict(X_scaled)
    )

    # ------------------------------------------------------------------------
    # Stability validation
    # ------------------------------------------------------------------------

    (
        stability_df,
        average_ari,
        average_silhouette,
        stability_level,
    ) = validate_cluster_stability(
        X_scaled,
        reference_labels,
    )

    # ------------------------------------------------------------------------
    # Final business segmentation
    # ------------------------------------------------------------------------

    (
        df,
        cluster_profile,
        final_kmeans,
    ) = create_final_clusters(
        df,
        X_scaled,
    )

    # ------------------------------------------------------------------------
    # Assign business labels
    # ------------------------------------------------------------------------

    (
        df,
        cluster_profile,
        segment_mapping,
    ) = assign_business_segments(
        df,
        cluster_profile,
    )

    # ------------------------------------------------------------------------
    # Final K=4 metrics
    # ------------------------------------------------------------------------

    final_row = (
        validation_metrics[
            validation_metrics["k"]
            == FINAL_K
        ]
        .iloc[0]
    )

    final_silhouette = (
        final_row[
            "silhouette_score"
        ]
    )

    final_calinski = (
        final_row[
            "calinski_harabasz_score"
        ]
    )

    final_davies = (
        final_row[
            "davies_bouldin_score"
        ]
    )

    # ------------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------------

    save_outputs(
        df,
        cluster_profile,
        final_kmeans,
        scaler,
        validation_metrics,
        stability_df,
    )

    # ------------------------------------------------------------------------
    # Model selection decision
    # ------------------------------------------------------------------------

    best_silhouette_k = int(
        validation_metrics.loc[
            validation_metrics[
                "silhouette_score"
            ].idxmax(),
            "k",
        ]
    )

    best_silhouette_score = (
        validation_metrics[
            "silhouette_score"
        ]
        .max()
    )

    print()
    print("=" * 80)
    print("MODEL SELECTION DECISION")
    print("=" * 80)

    print(
        f"Highest silhouette K: "
        f"{best_silhouette_k}"
    )

    print(
        f"Highest silhouette score: "
        f"{best_silhouette_score:.4f}"
    )

    print(
        f"Selected business K: "
        f"{FINAL_K}"
    )

    print(
        "\nReason for selecting K=4:"
    )

    print(
        "K=2 provides the strongest mathematical "
        "separation according to silhouette score."
    )

    print(
        "K=4 is retained because it produces "
        "four actionable customer groups with "
        "distinct RFM business characteristics."
    )

    print(
        "K=4 stability is validated separately "
        "using multiple random seeds and "
        "Adjusted Rand Index."
    )

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "RFM CLUSTERING VALIDATION & REFINEMENT COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nCustomers segmented: "
        f"{len(df):,}"
    )

    print(
        f"Final business clusters: "
        f"{FINAL_K}"
    )

    print(
        f"\nFinal K=4 validation:"
    )

    print(
        f"Silhouette Score: "
        f"{final_silhouette:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{final_calinski:,.2f}"
    )

    print(
        f"Davies-Bouldin Score: "
        f"{final_davies:.4f}"
    )

    print(
        f"\nAverage K=4 ARI: "
        f"{average_ari:.4f}"
    )

    print(
        f"Average K=4 silhouette: "
        f"{average_silhouette:.4f}"
    )

    print(
        f"Cluster stability: "
        f"{stability_level}"
    )

    print(
        "\nBusiness segments:"
    )

    for segment in sorted(
        cluster_profile[
            "segment"
        ].unique()
    ):

        row = cluster_profile[
            cluster_profile[
                "segment"
            ]
            == segment
        ].iloc[0]

        print(
            f"  {segment}: "
            f"{int(row['customers']):,} customers "
            f"({row['customer_percentage']:.2f}%)"
        )

    print(
        "\nNext stage:"
    )

    print(
        "Use the validated customer segments "
        "for downstream business recommendations, "
        "customer targeting, visualization, and "
        "final project reporting."
    )

    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":

    main()
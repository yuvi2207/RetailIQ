import os
import pandas as pd


# ============================================================================
# RetailIQ - RFM Business Recommendations
# Step 3: Convert validated customer segments into business strategies
#
# IMPORTANT:
# Segment names are taken from cluster_profile.csv rather than hard-coded
# cluster IDs. This prevents cluster-label swapping when K-Means is rerun.
# ============================================================================


OUTPUT_DIR = "outputs"

CUSTOMER_FILE = os.path.join(
    OUTPUT_DIR,
    "customer_segments.csv"
)

PROFILE_FILE = os.path.join(
    OUTPUT_DIR,
    "cluster_profile.csv"
)

STRATEGY_FILE = os.path.join(
    OUTPUT_DIR,
    "rfm_segment_strategy.csv"
)

RECOMMENDATION_FILE = os.path.join(
    OUTPUT_DIR,
    "rfm_business_recommendations.csv"
)


# ============================================================================
# Load data
# ============================================================================

def load_data():

    print("\nLoading customer segments...")

    customers = pd.read_csv(
        CUSTOMER_FILE
    )

    print(
        f"Customers loaded: {len(customers):,}"
    )

    print("\nLoading cluster profile...")

    profile = pd.read_csv(
        PROFILE_FILE
    )

    print(
        f"Segments loaded: {len(profile)}"
    )

    return customers, profile


# ============================================================================
# Validate input
# ============================================================================

def validate_data(customers, profile):

    required_customer_columns = [
        "customer_unique_id",
        "cluster",
        "segment",
        "recency",
        "frequency",
        "monetary"
    ]

    required_profile_columns = [
        "segment",
        "customers",
        "customer_percentage",
        "avg_recency",
        "avg_frequency",
        "avg_monetary"
    ]

    missing_customer = [
        col
        for col in required_customer_columns
        if col not in customers.columns
    ]

    missing_profile = [
        col
        for col in required_profile_columns
        if col not in profile.columns
    ]

    if missing_customer:

        raise ValueError(
            "Missing columns in customer_segments.csv: "
            + ", ".join(missing_customer)
        )

    if missing_profile:

        raise ValueError(
            "Missing columns in cluster_profile.csv: "
            + ", ".join(missing_profile)
        )

    if customers["segment"].isnull().any():

        raise ValueError(
            "Some customers do not have a segment."
        )

    if profile["segment"].isnull().any():

        raise ValueError(
            "Some clusters do not have a segment."
        )

    print("\nInput validation passed.")


# ============================================================================
# IMPORTANT FIX
# ============================================================================
#
# Do NOT create a mapping such as:
#
#     0 = High Value
#     1 = At Risk
#
# because K-Means cluster numbers are arbitrary.
#
# Instead, cluster_profile.csv and customer_segments.csv are treated as the
# source of truth for the current validated segmentation.
# ============================================================================


def synchronize_segment_labels(customers, profile):

    print("\nSynchronizing segment labels...")

    # Build cluster -> segment mapping from the already validated
    # customer_segments.csv.

    cluster_segment_map = (
        customers[
            ["cluster", "segment"]
        ]
        .drop_duplicates()
    )

    # Every cluster should have exactly one segment.

    duplicate_cluster_labels = (
        cluster_segment_map
        .groupby("cluster")["segment"]
        .nunique()
    )

    if (duplicate_cluster_labels > 1).any():

        raise ValueError(
            "A cluster has multiple segment labels. "
            "Segmentation output is inconsistent."
        )

    # Create a clean profile from the customer-level data.

    profile_from_customers = (
        customers
        .groupby("segment")
        .agg(
            customers=(
                "customer_unique_id",
                "count"
            ),
            avg_recency=(
                "recency",
                "mean"
            ),
            avg_frequency=(
                "frequency",
                "mean"
            ),
            avg_monetary=(
                "monetary",
                "mean"
            ),
            median_recency=(
                "recency",
                "median"
            ),
            median_frequency=(
                "frequency",
                "median"
            ),
            median_monetary=(
                "monetary",
                "median"
            )
        )
        .reset_index()
    )

    profile_from_customers[
        "customer_percentage"
    ] = (
        profile_from_customers["customers"]
        / len(customers)
        * 100
    )

    profile_from_customers = (
        profile_from_customers
        .round(2)
    )

    # Check that profile and customer-level segmentation agree.

    profile_check = (
        profile[
            [
                "segment",
                "customers",
                "avg_recency",
                "avg_frequency",
                "avg_monetary"
            ]
        ]
        .copy()
    )

    merged = profile_check.merge(
        profile_from_customers[
            [
                "segment",
                "customers",
                "avg_recency",
                "avg_frequency",
                "avg_monetary"
            ]
        ],
        on="segment",
        suffixes=(
            "_profile",
            "_customers"
        )
    )

    if len(merged) != len(profile_check):

        raise ValueError(
            "cluster_profile.csv and customer_segments.csv "
            "contain different segment labels."
        )

    print("\nValidated segment mapping:")

    print(
        profile_from_customers[
            [
                "segment",
                "customers",
                "customer_percentage",
                "avg_recency",
                "avg_frequency",
                "avg_monetary"
            ]
        ]
        .sort_values(
            "avg_monetary",
            ascending=False
        )
        .to_string(index=False)
    )

    return profile_from_customers


# ============================================================================
# Business strategy definitions
# ============================================================================

def create_strategy():

    strategies = {

        "Loyal High Value": {
            "business_objective": "Retain",
            "priority": "Very High",
            "risk_level": "Medium",

            "recommended_action": (
                "Protect loyalty and increase long-term customer value."
            ),

            "campaign": (
                "VIP loyalty programs, exclusive access, "
                "personalized rewards, early product access "
                "and cross-selling."
            ),

            "avoid": (
                "Heavy blanket discounts that reduce margin unnecessarily."
            )
        },

        "High Value": {
            "business_objective": "Retain / Reactivate",
            "priority": "Very High",
            "risk_level": "Medium",

            "recommended_action": (
                "Protect high-value customers and encourage "
                "their next purchase before inactivity increases."
            ),

            "campaign": (
                "Personalized product recommendations, "
                "targeted reactivation campaigns, premium offers "
                "and loyalty incentives."
            ),

            "avoid": (
                "Treating valuable customers with generic mass promotions."
            )
        },

        "Recent Low Value": {
            "business_objective": "Convert",
            "priority": "High",
            "risk_level": "Low",

            "recommended_action": (
                "Convert recent low-value or first-time customers "
                "into repeat buyers."
            ),

            "campaign": (
                "Second-purchase incentives, complementary products, "
                "bundles and personalized recommendations."
            ),

            "avoid": (
                "Large discounts for every customer when a smaller "
                "incentive may be sufficient."
            )
        },

        "At Risk": {
            "business_objective": "Win Back",
            "priority": "Very High",
            "risk_level": "High",

            "recommended_action": (
                "Recover customers who have gone a long period "
                "without purchasing."
            ),

            "campaign": (
                "Win-back campaigns, personalized reminders, "
                "targeted incentives and relevant product recommendations."
            ),

            "avoid": (
                "Repeated untargeted discounts without measuring "
                "reactivation."
            )
        }
    }

    return strategies


# ============================================================================
# Calculate revenue metrics
# ============================================================================

def calculate_revenue_metrics(customers):

    print("\nCalculating business metrics...")

    revenue = (
        customers
        .groupby("segment")
        .agg(
            customers=(
                "customer_unique_id",
                "count"
            ),
            total_revenue=(
                "monetary",
                "sum"
            ),
            average_customer_value=(
                "monetary",
                "mean"
            )
        )
        .reset_index()
    )

    total_revenue = revenue[
        "total_revenue"
    ].sum()

    revenue["revenue_percentage"] = (
        revenue["total_revenue"]
        / total_revenue
        * 100
    )

    revenue = revenue.round(2)

    return revenue


# ============================================================================
# Create final business strategy table
# ============================================================================

def build_strategy_table(profile, revenue, strategies):

    rows = []

    for _, row in profile.iterrows():

        segment = row["segment"]

        if segment not in strategies:

            raise ValueError(
                f"Unknown segment '{segment}'. "
                f"Expected one of: "
                f"{list(strategies.keys())}"
            )

        strategy = strategies[segment]

        revenue_row = revenue[
            revenue["segment"] == segment
        ]

        if revenue_row.empty:

            raise ValueError(
                f"No revenue data found for segment: {segment}"
            )

        revenue_row = revenue_row.iloc[0]

        rows.append({

            "segment": segment,

            "customers": int(
                row["customers"]
            ),

            "customer_percentage": round(
                row["customer_percentage"],
                2
            ),

            "avg_recency": round(
                row["avg_recency"],
                2
            ),

            "avg_frequency": round(
                row["avg_frequency"],
                2
            ),

            "avg_monetary": round(
                row["avg_monetary"],
                2
            ),

            "total_revenue": round(
                revenue_row["total_revenue"],
                2
            ),

            "revenue_percentage": round(
                revenue_row["revenue_percentage"],
                2
            ),

            "business_objective": (
                strategy["business_objective"]
            ),

            "priority": strategy["priority"],

            "risk_level": strategy["risk_level"],

            "recommended_action": (
                strategy["recommended_action"]
            ),

            "campaign": strategy["campaign"],

            "avoid": strategy["avoid"]
        })

    return pd.DataFrame(rows)


# ============================================================================
# Print business strategy
# ============================================================================

def print_business_strategy(strategy_df):

    print("\n")
    print("=" * 80)
    print("BUSINESS SEGMENT STRATEGY")
    print("=" * 80)

    # Print in business priority order.

    priority_order = {
        "Very High": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    display_df = (
        strategy_df
        .assign(
            _priority_order=
            strategy_df["priority"]
            .map(priority_order)
        )
        .sort_values(
            "_priority_order"
        )
    )

    for _, row in display_df.iterrows():

        print("\n")
        print(row["segment"])
        print("-" * 60)

        print(
            f"Customers: "
            f"{row['customers']:,} "
            f"({row['customer_percentage']:.2f}%)"
        )

        print(
            f"Average recency: "
            f"{row['avg_recency']:.2f} days"
        )

        print(
            f"Average frequency: "
            f"{row['avg_frequency']:.2f}"
        )

        print(
            f"Average monetary value: "
            f"₹{row['avg_monetary']:,.2f}"
        )

        print(
            f"Revenue contribution: "
            f"{row['revenue_percentage']:.2f}%"
        )

        print(
            f"Business objective: "
            f"{row['business_objective']}"
        )

        print(
            f"Priority: "
            f"{row['priority']}"
        )

        print(
            f"Risk level: "
            f"{row['risk_level']}"
        )

        print("\nRecommended action:")

        print(
            f"  {row['recommended_action']}"
        )

        print("\nCampaign:")

        print(
            f"  {row['campaign']}"
        )

        print("\nAvoid:")

        print(
            f"  {row['avoid']}"
        )


# ============================================================================
# Executive summary
# ============================================================================

def print_executive_summary(strategy_df):

    largest_customer_segment = (
        strategy_df
        .loc[
            strategy_df["customers"].idxmax()
        ]
    )

    largest_revenue_segment = (
        strategy_df
        .loc[
            strategy_df["total_revenue"].idxmax()
        ]
    )

    highest_value_segment = (
        strategy_df
        .loc[
            strategy_df["avg_monetary"].idxmax()
        ]
    )

    highest_recency_segment = (
        strategy_df
        .loc[
            strategy_df["avg_recency"].idxmax()
        ]
    )

    print("\n")
    print("=" * 80)
    print("EXECUTIVE BUSINESS SUMMARY")
    print("=" * 80)

    print("\nLargest customer segment:")

    print(
        f"  {largest_customer_segment['segment']} "
        f"({largest_customer_segment['customer_percentage']:.2f}% of customers)"
    )

    print("\nLargest revenue contributor:")

    print(
        f"  {largest_revenue_segment['segment']} "
        f"({largest_revenue_segment['revenue_percentage']:.2f}% of revenue)"
    )

    print("\nHighest average monetary value:")

    print(
        f"  {highest_value_segment['segment']} "
        f"(₹{highest_value_segment['avg_monetary']:,.2f})"
    )

    print("\nHighest recency / inactivity:")

    print(
        f"  {highest_recency_segment['segment']} "
        f"({highest_recency_segment['avg_recency']:.2f} days)"
    )

    print("\nKey strategic priorities:")

    print(
        "  1. Protect high-value and loyal customers."
    )

    print(
        "  2. Reactivate valuable inactive customers."
    )

    print(
        "  3. Convert recent low-value customers into repeat buyers."
    )

    print(
        "  4. Run targeted win-back campaigns for At Risk customers."
    )


# ============================================================================
# Save outputs
# ============================================================================

def save_outputs(strategy_df):

    strategy_df.to_csv(
        STRATEGY_FILE,
        index=False
    )

    strategy_df[
        [
            "segment",
            "business_objective",
            "priority",
            "risk_level",
            "recommended_action",
            "campaign",
            "avoid"
        ]
    ].to_csv(
        RECOMMENDATION_FILE,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"\n1. {os.path.abspath(STRATEGY_FILE)}"
    )

    print(
        f"2. {os.path.abspath(RECOMMENDATION_FILE)}"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 80)
    print("RETAILIQ - RFM BUSINESS RECOMMENDATIONS")
    print("=" * 80)

    customers, profile = load_data()

    validate_data(
        customers,
        profile
    )

    profile = synchronize_segment_labels(
        customers,
        profile
    )

    revenue = calculate_revenue_metrics(
        customers
    )

    strategies = create_strategy()

    print("\nCreating segment strategies...")

    strategy_df = build_strategy_table(
        profile,
        revenue,
        strategies
    )

    print_business_strategy(
        strategy_df
    )

    print_executive_summary(
        strategy_df
    )

    save_outputs(
        strategy_df
    )

    print("\n")
    print("=" * 80)
    print("RFM BUSINESS RECOMMENDATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
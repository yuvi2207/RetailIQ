import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# RetailIQ - RFM Customer Segmentation
# Step 2: Customer Segment Visualization & Business Analysis
# ============================================================================


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "customer_segments.csv"
)

PROFILE_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "cluster_profile.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "rfm_visuals"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():

    print("=" * 80)
    print("RETAILIQ - RFM CUSTOMER SEGMENT VISUALIZATION")
    print("=" * 80)

    print("\nLoading customer segments...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Customers loaded: {len(df):,}")

    print("\nLoading cluster profile...")

    profile = pd.read_csv(PROFILE_FILE)

    print(f"Segments loaded: {len(profile)}")

    return df, profile


# ---------------------------------------------------------------------------
# Prepare segment order
# ---------------------------------------------------------------------------

def prepare_data(df, profile):

    segment_order = [
        "Loyal High Value",
        "High Value",
        "Recent Low Value",
        "At Risk"
    ]

    df["segment"] = pd.Categorical(
        df["segment"],
        categories=segment_order,
        ordered=True
    )

    profile["segment"] = pd.Categorical(
        profile["segment"],
        categories=segment_order,
        ordered=True
    )

    profile = profile.sort_values("segment")

    return df, profile


# ---------------------------------------------------------------------------
# Segment distribution
# ---------------------------------------------------------------------------

def plot_segment_distribution(profile):

    print("\nCreating segment distribution chart...")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        profile["segment"],
        profile["customers"]
    )

    ax.set_title(
        "RetailIQ Customer Segment Distribution",
        fontsize=16
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")

    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(
        bars,
        profile["customers"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "segment_distribution.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Segment percentage
# ---------------------------------------------------------------------------

def plot_segment_percentage(profile):

    print("\nCreating segment percentage chart...")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        profile["segment"],
        profile["customer_percentage"]
    )

    ax.set_title(
        "Customer Distribution by Segment",
        fontsize=16
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Customers (%)")

    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(
        bars,
        profile["customer_percentage"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "segment_percentage.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Average RFM profile
# ---------------------------------------------------------------------------

def plot_rfm_profile(profile):

    print("\nCreating RFM profile charts...")

    metrics = [
        ("avg_recency", "Average Recency (Days)", "Lower is better"),
        ("avg_frequency", "Average Frequency", "Higher is better"),
        ("avg_monetary", "Average Monetary Value", "Higher is better")
    ]

    for column, ylabel, note in metrics:

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            profile["segment"],
            profile[column]
        )

        ax.set_title(
            f"{ylabel} by Customer Segment",
            fontsize=16
        )

        ax.set_xlabel("Customer Segment")
        ax.set_ylabel(ylabel)

        ax.tick_params(axis="x", rotation=20)

        for bar, value in zip(
            bars,
            profile[column]
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,.2f}",
                ha="center",
                va="bottom"
            )

        ax.text(
            0.99,
            0.95,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9
        )

        plt.tight_layout()

        filename = column.replace("avg_", "") + "_by_segment.png"

        output = os.path.join(
            OUTPUT_DIR,
            filename
        )

        plt.savefig(output, dpi=150)
        plt.close()

        print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Recency vs Monetary
# ---------------------------------------------------------------------------

def plot_recency_monetary(df):

    print("\nCreating Recency vs Monetary scatter plot...")

    # Sample to keep visualization lightweight
    sample_size = min(15000, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for segment in df["segment"].cat.categories:

        segment_data = sample[
            sample["segment"] == segment
        ]

        if len(segment_data) == 0:
            continue

        ax.scatter(
            segment_data["recency"],
            segment_data["monetary_log"],
            label=segment,
            alpha=0.35,
            s=15
        )

    ax.set_title(
        "Customer Segments: Recency vs Monetary Value",
        fontsize=16
    )

    ax.set_xlabel("Recency (Days)")
    ax.set_ylabel("Log(Monetary Value + 1)")

    ax.legend()

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "recency_vs_monetary.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Recency vs Frequency
# ---------------------------------------------------------------------------

def plot_recency_frequency(df):

    print("\nCreating Recency vs Frequency scatter plot...")

    sample_size = min(15000, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for segment in df["segment"].cat.categories:

        segment_data = sample[
            sample["segment"] == segment
        ]

        if len(segment_data) == 0:
            continue

        ax.scatter(
            segment_data["recency"],
            segment_data["frequency"],
            label=segment,
            alpha=0.35,
            s=15
        )

    ax.set_title(
        "Customer Segments: Recency vs Frequency",
        fontsize=16
    )

    ax.set_xlabel("Recency (Days)")
    ax.set_ylabel("Purchase Frequency")

    ax.legend()

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "recency_vs_frequency.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Segment revenue contribution
# ---------------------------------------------------------------------------

def calculate_segment_revenue(df):

    print("\nCalculating segment revenue contribution...")

    revenue = (
        df.groupby("segment", observed=True)["monetary"]
        .agg(
            customers="count",
            total_revenue="sum",
            average_customer_value="mean"
        )
        .reset_index()
    )

    revenue["revenue_percentage"] = (
        revenue["total_revenue"]
        / revenue["total_revenue"].sum()
        * 100
    )

    revenue = revenue.round(2)

    output = os.path.join(
        OUTPUT_DIR,
        "segment_revenue_analysis.csv"
    )

    revenue.to_csv(
        output,
        index=False
    )

    print("\nSegment Revenue Analysis:")
    print(revenue.to_string(index=False))

    print(f"\nSaved: {output}")

    return revenue


# ---------------------------------------------------------------------------
# Revenue visualization
# ---------------------------------------------------------------------------

def plot_segment_revenue(revenue):

    print("\nCreating segment revenue chart...")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        revenue["segment"],
        revenue["total_revenue"]
    )

    ax.set_title(
        "Revenue Contribution by Customer Segment",
        fontsize=16
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Total Revenue")

    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(
        bars,
        revenue["total_revenue"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"₹{value:,.0f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "segment_revenue.png"
    )

    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Business interpretation
# ---------------------------------------------------------------------------

def print_business_insights(profile, revenue):

    print("\n")
    print("=" * 80)
    print("BUSINESS INSIGHTS")
    print("=" * 80)

    for _, row in profile.iterrows():

        segment = row["segment"]

        customers = int(row["customers"])
        percentage = row["customer_percentage"]
        recency = row["avg_recency"]
        frequency = row["avg_frequency"]
        monetary = row["avg_monetary"]

        segment_revenue = revenue[
            revenue["segment"] == segment
        ]

        revenue_pct = segment_revenue[
            "revenue_percentage"
        ].iloc[0]

        print(f"\n{segment}")
        print("-" * 50)

        print(
            f"Customers: {customers:,} "
            f"({percentage:.2f}%)"
        )

        print(
            f"Average recency: {recency:.2f} days"
        )

        print(
            f"Average frequency: {frequency:.2f}"
        )

        print(
            f"Average monetary value: ₹{monetary:,.2f}"
        )

        print(
            f"Revenue contribution: "
            f"{revenue_pct:.2f}%"
        )

    print("\n")
    print("=" * 80)
    print("RECOMMENDED BUSINESS ACTIONS")
    print("=" * 80)

    print("""
1. Loyal High Value
   - Prioritize retention.
   - Use loyalty rewards and exclusive offers.
   - Avoid unnecessary discounting.

2. High Value
   - These customers spend heavily but are less recent.
   - Use targeted reactivation campaigns.
   - Encourage their next purchase before they become At Risk.

3. Recent Low Value
   - Large customer base with recent activity.
   - Focus on second-purchase conversion.
   - Recommend complementary products and bundles.

4. At Risk
   - Customers have relatively high recency.
   - Use win-back campaigns.
   - Test personalized incentives and reminders.
""")


# ---------------------------------------------------------------------------
# Save final analytical summary
# ---------------------------------------------------------------------------

def save_summary(profile, revenue):

    summary = profile.merge(
        revenue[
            [
                "segment",
                "total_revenue",
                "revenue_percentage",
                "average_customer_value"
            ]
        ],
        on="segment",
        how="left"
    )

    output = os.path.join(
        OUTPUT_DIR,
        "rfm_business_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(f"\nSaved final summary: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    df, profile = load_data()

    df, profile = prepare_data(
        df,
        profile
    )

    # Visualizations
    plot_segment_distribution(profile)

    plot_segment_percentage(profile)

    plot_rfm_profile(profile)

    plot_recency_monetary(df)

    plot_recency_frequency(df)

    # Revenue analysis
    revenue = calculate_segment_revenue(df)

    plot_segment_revenue(revenue)

    # Business interpretation
    print_business_insights(
        profile,
        revenue
    )

    # Final summary
    save_summary(
        profile,
        revenue
    )

    print("\n")
    print("=" * 80)
    print("RFM VISUALIZATION & BUSINESS ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nGenerated files:")
    print("outputs/rfm_visuals/segment_distribution.png")
    print("outputs/rfm_visuals/segment_percentage.png")
    print("outputs/rfm_visuals/recency_by_segment.png")
    print("outputs/rfm_visuals/frequency_by_segment.png")
    print("outputs/rfm_visuals/monetary_by_segment.png")
    print("outputs/rfm_visuals/recency_vs_monetary.png")
    print("outputs/rfm_visuals/recency_vs_frequency.png")
    print("outputs/rfm_visuals/segment_revenue.png")
    print("outputs/rfm_visuals/segment_revenue_analysis.csv")
    print("outputs/rfm_visuals/rfm_business_summary.csv")


if __name__ == "__main__":
    main()
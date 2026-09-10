from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/olist")

reviews = pd.read_csv(
    DATA_DIR / "olist_order_reviews_dataset.csv"
)

duplicates = reviews[
    reviews.duplicated("review_id", keep=False)
].sort_values("review_id")

print("=" * 80)
print("RETAILIQ - DUPLICATE REVIEW INVESTIGATION")
print("=" * 80)

print(f"\nTotal review rows: {len(reviews):,}")
print(f"Unique review IDs: {reviews['review_id'].nunique():,}")
print(f"Duplicate review rows: {len(duplicates):,}")

print("\nDuplicate review IDs:")
print(
    duplicates[
        [
            "review_id",
            "order_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("DUPLICATE REVIEW ID SUMMARY")
print("=" * 80)

summary = (
    duplicates
    .groupby("review_id")
    .agg(
        rows=("review_id", "size"),
        unique_orders=("order_id", "nunique"),
        unique_scores=("review_score", "nunique"),
        unique_titles=("review_comment_title", "nunique"),
        unique_messages=("review_comment_message", "nunique"),
    )
)

print(summary.to_string())

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
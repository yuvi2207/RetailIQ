from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/olist")

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("RETAILIQ - OLIST DATASET INSPECTION")
print("=" * 80)

print(f"\nCSV files found: {len(csv_files)}\n")

for file in csv_files:
    print("\n" + "=" * 80)
    print(f"FILE: {file.name}")
    print("=" * 80)

    df = pd.read_csv(file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  None")
    else:
        for column, count in missing.items():
            percentage = (count / len(df)) * 100
            print(f"  - {column}: {count:,} ({percentage:.2f}%)")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
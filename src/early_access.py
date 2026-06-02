from datetime import date
import polars as pl

# 1. Load the data
df = pl.read_parquet(
    "../data/openalex_mesh/work.parquet", columns=["type", "publication_date"]
)

# 2. Establish your static data collection checkpoint
snapshot_date = date(2026, 4, 2)

# 3. Calculate future-dated papers relative to the download timestamp
early_access_count = df.filter(pl.col("publication_date") > snapshot_date).height

print("--- Snapshot Early Access Audit ---")
print(f"Data Download Date: {snapshot_date}")
print(f"Early Access Papers (Published after download date): {early_access_count:,}")

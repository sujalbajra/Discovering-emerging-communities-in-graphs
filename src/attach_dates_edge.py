# Goal: Attach publication_date of the citing paper (src) to each edge.
# Result: (paper_id, cited_id, publication_date) — your timestamped edge list.
#
# Strategy: single left join — edges_final is left, work_clean is right.
# We only pull publication_date from work_clean, nothing else.
# Left join so any unmatched rows surface as nulls rather than silent drops.

import polars as pl
from pathlib import Path

DATA_DIR = Path("../data/openalex_mesh")
OUT_DIR = Path("../data/mesh_processed")

# ── 1. Build the date lookup — only two columns needed ───────────────────────
# No point scanning all 18 columns of work_clean just for one field.
# SHAPE: (31_787_206, 2)
date_lookup = (
    pl.scan_parquet(DATA_DIR / "work.parquet")
    .select(["paper_id", "publication_date"])
)

# ── 2. Join publication_date onto edge list ──────────────────────────────────
# SHAPE: (648_067_842, 3)
(
    pl.scan_parquet(DATA_DIR / "edge_list.parquet")
    .join(
        date_lookup,
        on="paper_id",
        how="left"
    )
    .sink_parquet(OUT_DIR / "edges_timestamped.parquet")
)

print("edges_timestamped.parquet saved")

# ── 3. Verify ────────────────────────────────────────────────────────────────
result = (
    pl.scan_parquet(OUT_DIR / "edges_timestamped.parquet")
    .select([
        pl.len().alias("total_rows"),
        pl.col("publication_date").is_null().sum().alias("null_dates"),
        pl.col("publication_date").min().alias("earliest"),
        pl.col("publication_date").max().alias("latest"),
    ])
    .collect(engine="streaming")
)

print(f"\nTotal rows:   {result['total_rows'].item():,}  (expected 648,067,842)")
print(f"Null dates:   {result['null_dates'].item():,}  (expected 0)")
print(f"Date range:   {result['earliest'].item()} -> {result['latest'].item()}")

import polars as pl
import pyarrow.parquet as pq

path = "../data/mesh_processed/edges_timestamped.parquet"
schema = pl.scan_parquet(path).collect_schema()

# Row count from metadata — no data loaded
num_rows = pq.read_metadata(path).num_rows
num_cols = len(schema)

print(f"\n{'='*50}")
print(f"FILE: {path}")
print(f"Shape: {num_rows:,} rows x {num_cols} cols")
print(f"{'─'*50}")
for col_name, dtype in schema.items():
    print(f"  {col_name:<35} {dtype}")

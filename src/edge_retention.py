import polars as pl
from pathlib import Path

DATA_DIR = Path("../data/openalex_mesh")

# ── 1. Total edge count ──────────────────────────────────────────────────────
# SHAPE: scalar — just counting rows, no data loaded into RAM
total_edges = (
    pl.scan_parquet(DATA_DIR / "cited_data.parquet")
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)

print(f"Total edges: {total_edges:,}")

# ── 2. Node set — just the paper_id column ───────────────────────────────────
# SHAPE: (31_787_247, 1) — lazy, not yet in RAM
node_set = (
    pl.scan_parquet(DATA_DIR / "work.parquet")
    .select("paper_id")
    .unique()
)

# ── 3. Semi-join to count within-dataset edges ───────────────────────────────
# A semi-join keeps rows from the LEFT table where a match exists in the RIGHT.
# It does NOT materialize the right table in full — Polars builds a hash set
# of the right key column and probes it row by row from the left.
#
# Memory cost: hash set of 31.8M string IDs ~ 2-3GB
# That's your peak RAM usage. Within your 11GB limit.
#
# SHAPE: (within_dataset_count, 2) — but we only collect the count
within_dataset = (
    pl.scan_parquet(DATA_DIR / "cited_data.parquet")
    .join(
        node_set,
        left_on="cited_id",
        right_on="paper_id",
        how="semi"            # keep left rows where cited_id is in node set
    )
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)

outside_dataset = total_edges - within_dataset
retention_pct   = within_dataset / total_edges * 100

print(f"\nWithin-dataset edges:  {within_dataset:,}  ({retention_pct:.1f}%)")
print(f"Outside-dataset edges: {outside_dataset:,}  ({100 - retention_pct:.1f}%)")

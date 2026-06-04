import polars as pl
from pathlib import Path

DATA_DIR = Path("../data/openalex_mesh")
OUT_DIR  = Path("../data/mesh_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

node_set = (
    pl.scan_parquet(DATA_DIR / "work.parquet")
    .select("paper_id")
    .unique()
)

# SHAPE: (648_067_949, 2) — streamed directly to disk, never in RAM
(
    pl.scan_parquet(DATA_DIR / "cited_data.parquet")
    .join(
        node_set,
        left_on="cited_id",
        right_on="paper_id",
        how="semi"
    )
    .sink_parquet(OUT_DIR / "edges_clean.parquet")
)

print("Done. Saved -> edges_clean.parquet")

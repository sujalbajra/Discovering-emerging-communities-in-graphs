import polars as pl
import pyarrow.parquet as pq
from pathlib import Path

data_dir = Path("../data/openalex_mesh/")

files = [
    "work.parquet",
    "cited_data.parquet",
    "keyword_data.parquet",
    "related_data.parquet",
    "mesh_data.parquet",
    "authors_data.parquet",
    "topic_data.parquet",
    "citation_counts_data.parquet",
    "prime_location_data.parquet",
    "descriptor_data.parquet"
]

for fname in files:
    path = data_dir / fname

    # Schema only — no data loaded
    schema = pl.scan_parquet(path).collect_schema()

    # Row count from metadata — no data loaded
    num_rows = pq.read_metadata(path).num_rows
    num_cols = len(schema)

    print(f"\n{'='*50}")
    print(f"FILE: {fname}")
    print(f"Shape: {num_rows:,} rows x {num_cols} cols")
    print(f"{'─'*50}")
    for col_name, dtype in schema.items():
        print(f"  {col_name:<35} {dtype}")

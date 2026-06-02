import polars as pl

(
    pl.scan_parquet("mesh_data.parquet")
    .select(
        pl.col("mesh")
          .explode()                          # one struct per row
          .struct.field("descriptor_name")    # pull just the name
          .alias("descriptor_name")
    )
    .group_by("descriptor_name")
    .agg(pl.len().alias("paper_count"))
    .sort("paper_count", descending=True)
    .collect(streaming=True)                  # streaming = bounded RAM
    .write_parquet("mesh_tag_counts.parquet") # persist for later
)

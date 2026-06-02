# import polars as pl

# mesh = pl.scan_parquet("../data/openalex_mesh/mesh_data.parquet").select("paper_id")
# work = pl.scan_parquet("../data/openalex_mesh/work.parquet").select("paper_id")

# mesh_ids = mesh.collect(engine="streaming").get_column("paper_id")
# work_ids = work.collect(engine="streaming").get_column("paper_id")

# mesh_set = set(mesh_ids)
# work_set = set(work_ids)

# in_mesh_not_work = mesh_set - work_set
# in_work_not_mesh = work_set - mesh_set

# print(f"mesh rows:              {len(mesh_set):>12,}")
# print(f"work rows:              {len(work_set):>12,}")
# print(f"in mesh, missing work:  {len(in_mesh_not_work):>12,}")
# print(f"in work, missing mesh:  {len(in_work_not_mesh):>12,}")
# print(f"overlap:                {len(mesh_set & work_set):>12,}")

# import polars as pl

# for name, path in [
#     ("mesh", "../data/openalex_mesh/mesh_data.parquet"),
#     ("work", "../data/openalex_mesh/work.parquet"),
# ]:
#     df = pl.scan_parquet(path).select("paper_id")
#     total = df.select(pl.len()).collect().item()
#     unique = df.select(pl.col("paper_id").n_unique()).collect().item()
#     print(f"{name:6s}  total={total:>12,}  unique={unique:>12,}  dupes={total-unique:>12,}")

import polars as pl

for name, path in [
    ("mesh", "../data/openalex_mesh/mesh_data.parquet"),
    ("work", "../data/openalex_mesh/work.parquet"),
]:
    dupes = (
        pl.scan_parquet(path)
        .select("paper_id")
        .group_by("paper_id")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .sort("count", descending=True)
        .collect(engine="streaming")
    )
    print(f"\n{name}: {len(dupes):,} duplicated paper_ids")
    print(dupes.head(20))

def save_duplicates():
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
        out = f"../data/mesh_processed/{name}_dupes.csv"
        dupes.write_csv(out)
        print(f"{name}: {len(dupes):,} dupes saved to {out}")


def count_distribution():
    import polars as pl

    for name in ["mesh", "work"]:
        dupes = pl.read_csv(f"../data/mesh_processed/{name}_dupes.csv")
        print(f"\n{name}:")
        print(dupes.group_by("count").agg(pl.len().alias("frequency")).sort("count"))


def dupe_similarity():
    import polars as pl

    for name, path in [
        ("mesh", "../data/openalex_mesh/mesh_data.parquet"),
        ("work", "../data/openalex_mesh/work.parquet"),
    ]:
        dupe_ids = (
            pl.read_csv(f"../data/mesh_processed/{name}_dupes.csv")
            .get_column("paper_id")
            .to_list()
            [:5]
        )

        rows = (
            pl.scan_parquet(path)
            .filter(pl.col("paper_id").is_in(dupe_ids))
            .collect(engine="streaming")
            .sort("paper_id")
        )

        rows.write_parquet(f"../data/mesh_processed/check_{name}_dupes.parquet")


def check_dupes():
    import polars as pl
    pl.read_parquet('../data/mesh_processed/check_mesh_dupes.parquet').with_columns(
        pl.col('descriptor_ids').list.join("|"),
        pl.col('is_major_topics').list.eval(pl.element().cast(pl.String)).list.join("|")
    ).write_csv('../data/mesh_processed/check_mesh_dupes.csv')


def delete_dupes():
    import polars as pl

    for name, path in [
        ("mesh", "../data/openalex_mesh/mesh_data.parquet"),
        ("work", "../data/openalex_mesh/work.parquet"),
    ]:
        dupe_ids = (
            pl.read_csv(f"../data/mesh_processed/{name}_dupes.csv")
            .get_column("paper_id")
            .to_list()
        )

        clean = (
            pl.scan_parquet(path)
            .filter(~pl.col("paper_id").is_in(dupe_ids))
        )

        if name == "work":
            deduped_dupes = (
                pl.scan_parquet(path)
                .filter(pl.col("paper_id").is_in(dupe_ids))
                .collect(engine="streaming")
                .sort("updated_date", descending=True)   # sort only ~900K rows, not 32M
                .unique(subset=["paper_id"], keep="first")
                .lazy()
            )
        else:
            deduped_dupes = (
                pl.scan_parquet(path)
                .filter(pl.col("paper_id").is_in(dupe_ids))
                .unique(subset=["paper_id"], keep="first")
            )

        (
            pl.concat([clean, deduped_dupes])
            .sink_parquet(f"../data/mesh_processed/{name}_deduped.parquet")
        )
        print(f"{name} done")

    # verify
    for name in ["mesh", "work"]:
        n = pl.scan_parquet(f"../data/mesh_processed/{name}_deduped.parquet").select(pl.len()).collect().item()
        print(f"{name}: {n:,}")


def count_duplicates():
    import polars as pl

    for name, path in [
        ("mesh", "../data/mesh_processed/mesh_deduped.parquet"),
        ("work", "../data/mesh_processed/work_deduped.parquet"),
    ]:
        df = pl.scan_parquet(path).select("paper_id")
        total = df.select(pl.len()).collect().item()
        unique = df.select(pl.col("paper_id").n_unique()).collect().item()
        print(f"{name:6s}  total={total:>12,}  unique={unique:>12,}  dupes={total-unique:>12,}")


count_duplicates()

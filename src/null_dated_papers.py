def null_analysis():
    # Goal: For the 41 null-dated papers, check how many times they appear
    # in the edge list as src (citing) and dst (cited).
    # The dst count is critical — high dst = hub node, dropping it matters.

    import polars as pl
    from pathlib import Path

    DATA_DIR = Path("../data/openalex_mesh")
    OUT_DIR = Path("../data/mesh_processed")

    # ── 1. Get the 41 null-dated paper IDs ──────────────────────────────────────
    # SHAPE: (41, 1)
    null_papers = (
        pl.scan_parquet(DATA_DIR / "work.parquet")
        .filter(pl.col("publication_date").is_null())
        .select("paper_id")
        .collect(engine="streaming")
    )

    print(f"Null-dated papers: {len(null_papers)}")

    null_ids = null_papers["paper_id"].to_list()  # 41 IDs — tiny, safe as a list

    # ── 2. Check appearance in edge list ────────────────────────────────────────
    # We scan edges_clean.parquet once and compute both counts in a single pass.
    # Two aggregations:
    #   - count of rows where paper_id (src) is in null_ids
    #   - count of rows where cited_id (dst) is in null_ids
    #
    # Why a single scan: edges_clean is 648M rows / 4.8GB.
    # Scanning twice would double the I/O cost unnecessarily.

    edges = pl.scan_parquet(DATA_DIR / "edge_list.parquet")

    # SHAPE: (<=41, 2) — one row per null paper that appears as src
    src_counts = (
        edges
        .filter(pl.col("paper_id").is_in(null_ids))
        .group_by("paper_id")
        .agg(pl.len().alias("out_degree"))
        .collect(engine="streaming")
    )

    # SHAPE: (<=41, 2) — one row per null paper that appears as dst
    dst_counts = (
        edges
        .filter(pl.col("cited_id").is_in(null_ids))
        .group_by("cited_id")
        .agg(pl.len().alias("in_degree"))
        .rename({"cited_id": "paper_id"})
        .collect(engine="streaming")
    )

    # ── 3. Join results together ─────────────────────────────────────────────────
    # Full outer join so papers with zero src or zero dst appearances still show up
    # SHAPE: (41, 3)
    result = (
        null_papers
        .join(src_counts, on="paper_id", how="left")
        .join(dst_counts, on="paper_id", how="left")
        .fill_null(0)  # papers not appearing in edges get count 0
        .sort("in_degree", descending=True)
    )

    print("\nNull-dated papers — edge appearances:")
    print(result)
    result.write_csv(OUT_DIR / "null_date_paper.csv")

    print(f"\nMax in-degree among null papers:  {result['in_degree'].max()}")
    print(f"Max out-degree among null papers: {result['out_degree'].max()}")
    print(f"Papers with in_degree > 100:      {(result['in_degree'] > 100).sum()}")
    print(f"Papers with zero edges:           {((result['in_degree'] == 0) & (result['out_degree'] == 0)).sum()}")

def drop_nulls():
    # Goal: Remove 41 null-dated papers from work and edges_clean.
    # Impact: negligible — max in-degree of 2, 29 are pure singletons.
    #
    # Two operations:
    #   1. Filter work.parquet — remove the 41 rows where publication_year is null
    #   2. Filter edges_clean.parquet — remove the 12 rows where cited_id is one of the 41

    import polars as pl
    from pathlib import Path

    DATA_DIR = Path("../data/openalex_mesh")
    OUT_DIR  = Path("../data/mesh_processed")

    # ── 1. Get the 41 null IDs ───────────────────────────────────────────────────
    # SHAPE: (41,)
    null_ids = (
        pl.scan_parquet(DATA_DIR / "work.parquet")
        .filter(pl.col("publication_date").is_null())
        .select("paper_id")
        .collect(engine="streaming")
        .get_column("paper_id")
        .to_list()
    )

    # ── 2. Clean work.parquet ────────────────────────────────────────────────────
    # SHAPE: (31_787_206, 18) — 41 rows removed
    (
        pl.scan_parquet(DATA_DIR / "work.parquet")
        .filter(pl.col("publication_date").is_not_null())
        .sink_parquet(OUT_DIR / "work_clean.parquet")
    )

    print("work_clean.parquet saved")

    # ── 3. Clean edges_clean.parquet ─────────────────────────────────────────────
    # Only cited_id needs filtering — out_degree was 0 for all 41 papers
    # meaning none of them appear as paper_id (src) in the edge list.
    # SHAPE: (648_067_937, 2) — 12 rows removed
    (
        pl.scan_parquet(DATA_DIR / "edge_list.parquet")
        .filter(~pl.col("cited_id").is_in(null_ids))
        .sink_parquet(OUT_DIR / "edges_final.parquet")
    )

    print("edges_final.parquet saved")

    # ── 4. Verify ────────────────────────────────────────────────────────────────
    work_count = (
        pl.scan_parquet(OUT_DIR / "work_clean.parquet")
        .select(pl.len())
        .collect()
        .item()
    )

    edge_count = (
        pl.scan_parquet(OUT_DIR / "edges_final.parquet")
        .select(pl.len())
        .collect()
        .item()
    )

    print(f"\nwork_clean rows:  {work_count:,}")
    print(f"edges_final rows: {edge_count:,}")

drop_nulls()

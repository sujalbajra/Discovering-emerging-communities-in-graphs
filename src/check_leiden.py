def print_head():
    import polars as pl

    # 1. Scan the file (creates a lazy pointer, doesn't load data yet)
    lazy_df = pl.scan_parquet("../data/mesh_processed/leiden_2006.parquet")

    # 2. Build the query and execute it
    df = lazy_df.limit(100).collect()

    # 3. Print the results
    print(df)

def check_size():
    import polars as pl

    # Your actual files: leiden_YYYY.parquet with columns paper_id (str), year (i32), community_cold (i32)
    # Note: "community_cold" — I'm assuming this is "community_id", confirm naming consistency
    # across files (typo risk: if some files say "community_id" and others "community_cold",
    # a naive pl.scan_parquet glob will silently produce nulls for mismatched columns)


    # Check schema consistency across ALL files first — cheap, critical
    for y in [1970, 1995, 2016]:  # spot check, not exhaustive
        schema = pl.scan_parquet(f"../data/mesh_processed/leiden/leiden_{y}.parquet").collect_schema()
        print(y, schema)

    # Diagnostic: community count and size distribution per year
    sizes = (
        pl.scan_parquet("../data/mesh_processed/leiden/leiden_*.parquet")  # glob — only works if schema is IDENTICAL across files
        .group_by(["year", "community_cold"])
        .agg(pl.len().alias("size"))
        .collect(streaming=True)
    )

    summary = (
        sizes.group_by("year")
        .agg(
            pl.len().alias("n_communities"),
            pl.col("size").mean().alias("mean_size"),
            pl.col("size").median().alias("median_size"),
            pl.col("size").max().alias("max_size"),
            pl.col("size").min().alias("min_size"),
        )
        .sort("year")
    )

    print(summary)

def check():
    import polars as pl

    # Check: how does community count near the giant component change?
    # i.e., is there ONE dominant community per year, or several?
    for y in [1970, 1990, 2010, 2016]:
        df = pl.scan_parquet(f"../data/mesh_processed/leiden/leiden_{y}.parquet")
        top10 = (
            df.group_by("community_cold")
            .agg(pl.len().alias("size"))
            .sort("size", descending=True)
            .head(10)
            .collect()
        )
        total_nodes = df.select(pl.len()).collect().item()
        print(f"\nYear {y} (total nodes: {total_nodes})")
        print(top10)
        print(f"Top community = {top10['size'][0] / total_nodes:.2%} of all nodes")

def percentile_diagnostic():
    import polars as pl

    for y in [1970, 1990, 2010, 2016]:
        sizes = (
            pl.scan_parquet(f"../data/mesh_processed/leiden/leiden_{y}.parquet")
            .group_by("community_cold")
            .agg(pl.len().alias("size"))
            .collect()
        )
        total_nodes = sizes["size"].sum()
        n_comm = sizes.height

        quantiles = sizes["size"].quantile(0.5), sizes["size"].quantile(0.75), \
                    sizes["size"].quantile(0.90), sizes["size"].quantile(0.95), \
                    sizes["size"].quantile(0.99)

        # Coverage: what fraction of NODES are in communities of size >= threshold?
        for thresh in [10, 50, 100, 500, 1000]:
            covered = sizes.filter(pl.col("size") >= thresh)["size"].sum()
            n_surviving = sizes.filter(pl.col("size") >= thresh).height
            print(f"Year {y} | thresh={thresh:>4} | "
                f"surviving comms = {n_surviving:>5}/{n_comm} | "
                f"node coverage = {covered/total_nodes:.2%}")

        print(f"Year {y} | p50={quantiles[0]}, p75={quantiles[1]}, "
            f"p90={quantiles[2]}, p95={quantiles[3]}, p99={quantiles[4]}")
        print()

def comm_count():
    from pathlib import Path
    import polars as pl

    column = "community_cold"

    for file in sorted(Path("../data/mesh_processed/leiden").glob("*.parquet")):
        n_unique = (
            pl.scan_parquet(file)
            .select(pl.col(column).n_unique())
            .collect()
            .item()
        )

        print(f"{file.name}: {n_unique}")

def comm_size_summary():
    from pathlib import Path
    import polars as pl

    column = "community_cold"

    for file in sorted(Path("../data/mesh_processed/leiden").glob("*.parquet")):
        # Group by community and count the number of members in each
        community_sizes = (
            pl.scan_parquet(file)
            .group_by(column)
            .agg(pl.len().alias("size"))
        )

        # Calculate summary statistics for the sizes
        summary = (
            community_sizes
            .select(
                pl.col("size").max().alias("largest"),
                pl.col("size").mean().alias("average"),
                pl.col("size").median().alias("median")
            )
            .collect()
        )

        # Extract values for clean printing
        largest = summary["largest"].item()
        average = round(summary["average"].item(), 1)
        median = summary["median"].item()

        print(f"{file.name} -> Largest: {largest} | Average: {average} | Median: {median}")

def comm_size_filtered_summary():
    from pathlib import Path
    import polars as pl

    column = "community_cold"

    for file in sorted(Path("../data/mesh_processed/leiden").glob("*.parquet")):
        # Group, count, and immediately filter out sizes 5 or smaller
        community_sizes = (
            pl.scan_parquet(file)
            .group_by(column)
            .agg(pl.len().alias("size"))
            .filter(pl.col("size") > 5)
        )

        # Calculate summary statistics on the filtered data
        summary = (
            community_sizes
            .select(
                pl.col("size").max().alias("largest"),
                pl.col("size").mean().alias("average"),
                pl.col("size").median().alias("median")
            )
            .collect()
        )

        # Extract values for clean printing
        largest = summary["largest"].item()
        # Handle cases where all communities might have been filtered out
        average = round(summary["average"].item(), 1) if summary["average"].item() is not None else 0
        median = summary["median"].item() if summary["median"].item() is not None else 0

        print(f"{file.name} -> Largest: {largest} | Average: {average} | Median: {median}")

def comm_size_bins():
    from pathlib import Path
    import polars as pl

    column = "community_cold"

    for file in sorted(Path("../data/mesh_processed/leiden").glob("*.parquet")):
        # Calculate sizes and filter out <= 5
        sizes = (
            pl.scan_parquet(file)
            .group_by(column)
            .agg(pl.len().alias("size"))
            .filter(pl.col("size") > 5)
        )

        # Categorize into bins
        binned = (
            sizes
            .with_columns(
                pl.when(pl.col("size") <= 50).then(pl.lit("Small (6-50)"))
                .when(pl.col("size") <= 1000).then(pl.lit("Medium (51-1k)"))
                .when(pl.col("size") <= 50000).then(pl.lit("Large (1k-50k)"))
                .otherwise(pl.lit("Mega (>50k)"))
                .alias("category")
            )
            .group_by("category")
            .agg(pl.len().alias("count"))
            .collect()
        )

        # Format output
        print(f"\n--- {file.name} ---")
        for row in binned.iter_rows(named=True):
            print(f"{row['category']}: {row['count']} communities")

def comm_size_granular_bins():
    from pathlib import Path
    import polars as pl

    column = "community_cold"

    for file in sorted(Path("../data/mesh_processed/leiden").glob("*.parquet")):
        # Calculate sizes and filter out <= 5
        sizes = (
            pl.scan_parquet(file)
            .group_by(column)
            .agg(pl.len().alias("size"))
            .filter(pl.col("size") > 5)
        )

        # Categorize into granular bins
        binned = (
            sizes
            .with_columns(
                pl.when(pl.col("size") <= 25).then(pl.lit("1. Micro (6-25)"))
                .when(pl.col("size") <= 100).then(pl.lit("2. Small (26-100)"))
                .when(pl.col("size") <= 1000).then(pl.lit("3. Medium (101-1k)"))
                .when(pl.col("size") <= 10000).then(pl.lit("4. Large (1k-10k)"))
                .when(pl.col("size") <= 100000).then(pl.lit("5. Jumbo (10k-100k)"))
                .when(pl.col("size") <= 500000).then(pl.lit("6. Mega (100k-500k)"))
                .otherwise(pl.lit("7. Giga (>500k)"))
                .alias("category")
            )
            .group_by("category")
            .agg(pl.len().alias("count"))
            .sort("category")
            .collect()
        )

        # Format output
        print(f"\n--- {file.name} ---")
        for row in binned.iter_rows(named=True):
            print(f"{row['category']}: {row['count']} communities")

def single_leiden_check():
    from pathlib import Path
    import polars as pl

    column = "community_cold"

    file = Path("../data/mesh_processed/leiden_1970.parquet")
    # Calculate sizes and filter out <= 5
    sizes = (
        pl.scan_parquet(file)
        .group_by(column)
        .agg(pl.len().alias("size"))
        .filter(pl.col("size") > 5)
    )

    # Categorize into granular bins
    binned = (
        sizes
        .with_columns(
            pl.when(pl.col("size") <= 25).then(pl.lit("1. Micro (6-25)"))
            .when(pl.col("size") <= 100).then(pl.lit("2. Small (26-100)"))
            .when(pl.col("size") <= 1000).then(pl.lit("3. Medium (101-1k)"))
            .when(pl.col("size") <= 10000).then(pl.lit("4. Large (1k-10k)"))
            .when(pl.col("size") <= 100000).then(pl.lit("5. Jumbo (10k-100k)"))
            .when(pl.col("size") <= 500000).then(pl.lit("6. Mega (100k-500k)"))
            .otherwise(pl.lit("7. Giga (>500k)"))
            .alias("category")
        )
        .group_by("category")
        .agg(pl.len().alias("count"))
        .sort("category")
        .collect()
    )

    # Format output
    print(f"\n--- {file.name} ---")
    for row in binned.iter_rows(named=True):
        print(f"{row['category']}: {row['count']} communities")

single_leiden_check()

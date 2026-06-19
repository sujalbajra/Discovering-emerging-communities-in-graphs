import polars as pl
import gc

results = []

for y in range(1970, 2017):
    edges = (
    pl.scan_parquet("/kaggle/input/datasets/sujalbajracharya/openalex-mesh-filtered-data-with-abstracts/edge_list.parquet")
    .filter(pl.col("publication_date").dt.year() <= y)
    .select(["paper_id", "cited_id"])
    )

    # n_edges: trivial, streaming-safe
    n_edges = edges.select(pl.len()).collect(engine="streaming").item()

    # n_nodes WITHOUT unpivot — avoids doubling row count before unique.
    # Get unique paper_ids and unique cited_ids separately (each streams fine),
    # then union via a single small collect on the unique SETS, not raw rows.
    unique_paper = (
        edges.select("paper_id").unique().rename({"paper_id": "node"})
    )
    unique_cited = (
        edges.select("cited_id").unique().rename({"cited_id": "node"})
    )

    n_nodes = (
        pl.concat([unique_paper, unique_cited])
        .unique()
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )

    density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    results.append({"year": y, "n_nodes": n_nodes, "n_edges": n_edges, "density": density})
    print(f"{y} | n_nodes={n_nodes:>10} | n_edges={n_edges:>10} | density={density:.3e}")

    # free intermediate frames between years
    del unique_paper, unique_cited
    gc.collect()

df = pl.DataFrame(results)
df.write_parquet("density_sweep.parquet")
print(df)

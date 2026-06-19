import polars as pl
import cudf
import cugraph
import numpy as np
import gc
import threading
import queue
import os
import rmm

# ── CRITICAL FIX: Enable Managed Memory (Unified Memory) ────────────────────
# This allows the GPU to use system RAM as a backup safety net during spikes,
# preventing the "out of memory" crash.
rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

# ── Step 1: Global integer node ID mapping ──────────────────────────────────
valid_ids = pl.scan_parquet("/kaggle/input/notebooks/rnewr013/openalex-mesh-22m/node_list.parquet")

id_map = (
    valid_ids
    .select("paper_id")
    .unique()
    .with_row_index(name="node_idx")
    .with_columns(pl.col("node_idx").cast(pl.Int32))
    .select(["paper_id", "node_idx"])
)
id_map.sink_parquet("id_map.parquet")
print("id_map built")

id_map_lazy = pl.scan_parquet("/kaggle/working/id_map.parquet")
id_map_df = id_map_lazy.collect(engine="streaming")
N_total = id_map_df.height
print(f"Total nodes in universe: {N_total}")

# ── Output directory for per-year files ─────────────────────────────────────
OUT_DIR = "/kaggle/working/leiden_per_year"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(1970, 2017))

# ── Writer thread setup ──────────────────────────────────────────────────────
write_queue = queue.Queue(maxsize=4)
SENTINEL = object()

def sanity_check_df(df, year):
    """Profiles the community sizes directly from memory, including tiny clusters."""
    column = "community_cold"

    # Calculate sizes for ALL communities (No filtering out <= 5)
    sizes = (
        df.lazy()
        .group_by(column)
        .agg(pl.len().alias("size"))
    )

    # Categorize into granular bins including the new Micro Pro tier
    binned = (
        sizes
        .with_columns(
            pl.when(pl.col("size") < 6).then(pl.lit("0. Micro Pro (<6)"))
            .when(pl.col("size") <= 25).then(pl.lit("1. Micro (6-25)"))
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
    print(f"\n--- [Sanity Check] Distribution for Year {year} ---")
    for row in binned.iter_rows(named=True):
        print(f"{row['category']}: {row['count']} communities")

def writer_worker():
    while True:
        item = write_queue.get()
        if item is SENTINEL:
            write_queue.task_done()
            break
        year, df = item

        # 1. Write to disk in the background
        path = os.path.join(OUT_DIR, f"leiden_{year}.parquet")
        df.write_parquet(path)
        print(f"  [writer] year {year} written -> {path}")

        # 2. Profile directly using the in-memory dataframe object
        sanity_check_df(df, year)

        write_queue.task_done()

writer_thread = threading.Thread(target=writer_worker, daemon=True)
writer_thread.start()

# ── Dynamic Resolution Function ─────────────────────────────────────────────
def get_scaled_resolution(current_node_count):
    # Anchor 1: 1970 baseline
    nodes_1970 = 203_410
    res_1970 = 5.0

    # Anchor 2: 2016 baseline
    nodes_2016 = 17_037_307
    res_2016 = 200.0

    # Linear interpolation
    slope = (res_2016 - res_1970) / (nodes_2016 - nodes_1970)
    scaled_res = res_1970 + slope * (current_node_count - nodes_1970)

    # Keep a safe minimum floor of 5.0
    return max(5.0, float(scaled_res))

# ── Main per-year loop ──────────────────────────────────────────────────────
for year in YEARS:

    # ── Stream-filter cumulative edges, map to int IDs ──────────────────────
    edges_year = (
        pl.scan_parquet("/kaggle/input/notebooks/rnewr013/openalex-mesh-22m/edge_list.parquet")
        .filter(pl.col("publication_date").dt.year() <= year)
        .select(["paper_id", "cited_id"])
    )

    edges_int_lazy = (
        edges_year
        .join(id_map_lazy, left_on="paper_id", right_on="paper_id", how="inner")
        .rename({"node_idx": "src_idx"})
        .join(id_map_lazy, left_on="cited_id", right_on="paper_id", how="inner")
        .rename({"node_idx": "dst_idx"})
        .select(["src_idx", "dst_idx"])
    )

    edges_int_df = edges_int_lazy.collect(engine="streaming")
    E_y = edges_int_df.height
    print(f"Year {year}: {E_y:,} edges")

    if E_y == 0:
        del edges_int_df
        gc.collect()
        continue

    # ── OPTIMIZED: Low-memory node count calculation ────────────────────────
    # Replaced heavy numpy array copies with a fast Polars unique count
    N_y = pl.concat([edges_int_df["src_idx"], edges_int_df["dst_idx"]]).n_unique()

    # ── Calculate Dynamic Resolution ──────────────────────────────────────────
    current_res = get_scaled_resolution(N_y)
    print(f"Year {year}: {N_y:,} nodes | Applied Resolution: {current_res:.2f}")

    # ── Convert edge list to cuDF for cuGraph ────────────────────────────────
    edges_arrow = edges_int_df.to_arrow()
    edges_gdf = cudf.DataFrame.from_arrow(edges_arrow)

    del edges_int_df, edges_int_lazy, edges_year
    gc.collect()

    # ── Build cuGraph Graph ───────────────────────────────────────────────────
    edges_gdf["weight"] = cudf.Series(np.ones(E_y, dtype=np.float32))

    G = cugraph.Graph(directed=False)

    # Force garbage collection right before the heaviest memory allocation
    gc.collect()

    G.from_cudf_edgelist(edges_gdf, source="src_idx", destination="dst_idx",
                          edge_attr="weight", renumber=True)

    del edges_gdf
    gc.collect()

    # free, total = rmm.mr.available_device_memory()
    # print(f"  GPU free: {free/1e9:.2f}GB / {total/1e9:.2f}GB")

    # ── Run Leiden ────────────────────────────────────────────────────────────
    parts, modularity = cugraph.leiden(
        G,
        max_iter=100,
        resolution=current_res,
        random_state=42,
    )
    print(f"Year {year}: modularity={modularity:.4f}")

    del G
    gc.collect()

    # ── Convert result back to Polars, join to paper_id strings ──────────────
    parts_pl = pl.from_arrow(parts.to_arrow())
    parts_pl = parts_pl.rename({"vertex": "node_idx", "partition": "community_cold"})
    parts_pl = parts_pl.with_columns(pl.lit(year, dtype=pl.Int32).alias("year"))

    out_df = (
        parts_pl
        .join(id_map_df, on="node_idx", how="left")
        .select(["paper_id", "year", "community_cold"])
    )

    del parts, parts_pl
    gc.collect()

    # ── Hand off to writer thread ────────────────────────────────────────────
    write_queue.put((year, out_df))

    del out_df
    gc.collect()

    print(f"Year {year}: queued for write, {N_y:,} nodes")

# ── Shutdown writer thread cleanly ───────────────────────────────────────────
print("Shutting down the writer")
write_queue.put(SENTINEL)
write_queue.join()
writer_thread.join()

print("All years complete.")

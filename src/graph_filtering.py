def clean_edges():
    """
    clean_edges_hashpart.py

    Removes self-citations and duplicate edges using hash-based partitioning.

    Input:  /kaggle/input/datasets/sujalbajracharya/temporary-dump/edge_list.parquet
    Output: /kaggle/working/edges_clean.parquet

    N_BUCKETS=8 — each bucket ~10M rows, ~3.7GB peak RAM per bucket
    """

    import polars as pl
    import pyarrow.parquet as pq
    from pathlib import Path
    import gc
    import time
    import shutil

    INPUT_PATH  = Path("/kaggle/input/datasets/sujalbajracharya/temporary-dump/edge_list.parquet")
    BUCKET_DIR  = Path("/kaggle/working/edge_buckets")
    OUTPUT_PATH = Path("/kaggle/working/edges_clean.parquet")

    BUCKET_DIR.mkdir(parents=True, exist_ok=True)

    N_BUCKETS = 8

    # ── Phase 1: Partition edges into buckets ─────────────────────────────────────
    print("Phase 1: Partitioning edges into buckets...")

    pf        = pq.ParquetFile(INPUT_PATH)
    n_rg      = pf.metadata.num_row_groups
    schema    = pf.schema_arrow
    t0        = time.time()
    total_in  = 0

    writers = {
        i: pq.ParquetWriter(BUCKET_DIR / f"bucket_{i:02d}.parquet", schema, compression="zstd")
        for i in range(N_BUCKETS)
    }

    for rg_idx in range(n_rg):
        # Read one row group — shape: (~123K, 2)
        batch = pf.read_row_group(rg_idx)
        df = pl.from_arrow(batch)
        # Shape: (~123K, 2) columns: [paper_id, cited_id]

        # Drop self-citations — shape: (~123K, 2) slightly fewer
        df = df.filter(pl.col("paper_id") != pl.col("cited_id"))

        # Assign bucket via hash of concatenated pair
        # hash() returns uint64 — mod N_BUCKETS gives bucket id
        # Shape: (~123K, 3) — added bucket column
        df = df.with_columns(
            ((pl.col("paper_id") + "|" + pl.col("cited_id"))
            .hash() % N_BUCKETS
            ).cast(pl.Int8).alias("bucket")
        )

        # Route to bucket writers
        for b in range(N_BUCKETS):
            subset = df.filter(pl.col("bucket") == b).drop("bucket")
            # Shape: (~123K/8 ≈ 15K rows per bucket per row group)
            if len(subset) > 0:
                writers[b].write_table(subset.to_arrow())

        total_in += len(df)

        if (rg_idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = total_in / elapsed / 1e6
            eta     = (n_rg - rg_idx - 1) * (elapsed / (rg_idx + 1))
            print(f"  [{rg_idx+1}/{n_rg}]  {total_in:,} edges  {rate:.2f}M rows/s  ETA {eta/60:.1f}min")

    for w in writers.values():
        w.close()

    elapsed = time.time() - t0
    print(f"Phase 1 done. {total_in:,} edges partitioned in {elapsed/60:.1f}min")
    del df, writers
    gc.collect()

    # ── Phase 2: Dedup within each bucket ────────────────────────────────────────
    # Each bucket ~80M rows (~3.7GB) — fits comfortably in 29GB RAM
    # Dedup is globally correct because duplicate pairs always hash to same bucket

    print("\nPhase 2: Deduplicating buckets...")

    deduped_paths = []
    total_clean   = 0
    total_removed = 0

    for b in range(N_BUCKETS):
        bucket_path  = BUCKET_DIR / f"bucket_{b:02d}.parquet"
        deduped_path = BUCKET_DIR / f"deduped_{b:02d}.parquet"

        # Shape: (~80M, 2)
        bucket_df = pl.read_parquet(bucket_path)
        before    = len(bucket_df)

        # Global dedup within bucket
        # Shape: (~80M, 2) — fewer rows
        bucket_df = bucket_df.unique(subset=["paper_id", "cited_id"], maintain_order=False)
        after     = len(bucket_df)

        bucket_df.write_parquet(deduped_path, compression="zstd")
        deduped_paths.append(deduped_path)

        removed       = before - after
        total_clean  += after
        total_removed += removed
        print(f"  Bucket {b:02d}: {before:,} → {after:,}  removed {removed:,}")

        del bucket_df
        gc.collect()

    print(f"\nPhase 2 done.")
    print(f"  Total removed:     {total_removed:,}")
    print(f"  Total clean edges: {total_clean:,}")
    # Expected: ~636,866,129

    # ── Phase 3: Merge deduped buckets into final output ─────────────────────────
    print("\nPhase 3: Merging into final output...")

    (
        pl.scan_parquet([str(p) for p in deduped_paths])
        # Shape: (total_clean, 2)
        .sink_parquet(OUTPUT_PATH, compression="zstd")
    )

    print(f"  Written: {OUTPUT_PATH}")

    # ── Phase 4: Cleanup ──────────────────────────────────────────────────────────
    shutil.rmtree(BUCKET_DIR)
    print("  Bucket directory removed.")

    # ── Sanity check ─────────────────────────────────────────────────────────────
    final_count = pl.scan_parquet(OUTPUT_PATH).select(pl.len()).collect().item()
    print(f"\nFinal edge count: {final_count:,}")
    print(f"Expected:         ~636,866,129")
    print(f"Difference:       {final_count - 636_866_129:,}")

def clean_nodes():
    """
    remove_singletons.py

    Identifies and removes singleton nodes from work.parquet.
    Uses uint64 hashing to avoid OOM on string columns.

    Peak RAM: ~211MB for connected hash set + one row group at a time

    Input:
    /kaggle/working/edges_clean.parquet
    /kaggle/input/datasets/sujalbajracharya/temporary-dump/work.parquet

    Output:
    /kaggle/working/work_clean.parquet      — non-singleton nodes, all 18 cols
    /kaggle/working/singleton_nodes.parquet — singleton paper_ids only
    """

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import xxhash
    import gc
    import time
    from pathlib import Path

    EDGES_PATH    = Path("/kaggle/working/edges_clean.parquet")
    WORK_PATH     = Path("/kaggle/input/datasets/sujalbajracharya/temporary-dump/work.parquet")
    WORK_OUT      = Path("/kaggle/working/work_clean.parquet")
    SINGLETON_OUT = Path("/kaggle/working/singleton_nodes.parquet")

    EDGE_RG_BATCH = 20   # row groups per read in phase 1
    WORK_RG_BATCH = 8    # row groups per read in phase 2

    def hash_col(str_list):
        """Hash a list of strings to uint64 using xxh64."""
        return np.array(
            [xxhash.xxh64_intdigest(s) for s in str_list],
            dtype=np.uint64
        )

    # ── Phase 1: Build connected node hash set from edges_clean ──────────────────
    # Stream edges in batches, hash both src and dst columns,
    # accumulate unique hashes into a set
    # Peak RAM: ~211MB for 26.4M uint64 hashes

    print("Phase 1: Building connected node hash set...")
    t0 = time.time()

    pf_edges  = pq.ParquetFile(EDGES_PATH)
    n_edge_rg = pf_edges.metadata.num_row_groups
    print(f"  Edge row groups: {n_edge_rg}")

    connected_hash_set = set()

    for start in range(0, n_edge_rg, EDGE_RG_BATCH):
        rg_batch = list(range(start, min(start + EDGE_RG_BATCH, n_edge_rg)))

        tbl = pa.concat_tables([
            pf_edges.read_row_group(rg, columns=["paper_id", "cited_id"])
            for rg in rg_batch
        ])
        # Shape: (~EDGE_RG_BATCH * 123K, 2)

        src_list = tbl.column("paper_id").to_pylist()
        dst_list = tbl.column("cited_id").to_pylist()
        del tbl

        # Hash both columns and add to set
        connected_hash_set.update(hash_col(src_list).tolist())
        connected_hash_set.update(hash_col(dst_list).tolist())
        del src_list, dst_list
        gc.collect()

        if (start // EDGE_RG_BATCH + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{start + EDGE_RG_BATCH}/{n_edge_rg} rg]  "
                f"set size: {len(connected_hash_set):,}  ({elapsed:.1f}s)")

    print(f"  Connected node hashes: {len(connected_hash_set):,}")
    print(f"  Phase 1 done ({time.time()-t0:.1f}s)")

    # ── Phase 2: Stream work.parquet, classify each node ─────────────────────────
    # Hash each paper_id, check membership in connected_hash_set
    # Route to work_clean or singletons writer

    print("\nPhase 2: Classifying nodes...")
    t0 = time.time()

    pf_work   = pq.ParquetFile(WORK_PATH)
    n_work_rg = pf_work.metadata.num_row_groups
    work_schema = pf_work.schema_arrow
    # Singleton output only needs paper_id
    singleton_schema = pa.schema([("paper_id", pa.string())])

    writer_clean     = pq.ParquetWriter(WORK_OUT,      work_schema,      compression="zstd")
    writer_singleton = pq.ParquetWriter(SINGLETON_OUT, singleton_schema, compression="zstd")

    total_clean     = 0
    total_singleton = 0

    for start in range(0, n_work_rg, WORK_RG_BATCH):
        rg_batch = list(range(start, min(start + WORK_RG_BATCH, n_work_rg)))

        tbl = pa.concat_tables([
            pf_work.read_row_group(rg)
            for rg in rg_batch
        ])
        # Shape: (~WORK_RG_BATCH * ~120K, 18)

        paper_ids = tbl.column("paper_id").to_pylist()

        # Hash each paper_id and check against connected set
        hashes     = hash_col(paper_ids)
        # Shape: (~960K,) uint64
        is_connected = np.array(
            [h in connected_hash_set for h in hashes.tolist()],
            dtype=bool
        )
        # Shape: (~960K,) bool

        # Split table using boolean mask
        connected_mask    = pa.array(is_connected)
        singleton_mask    = pa.array(~is_connected)

        clean_tbl     = tbl.filter(connected_mask)
        singleton_tbl = tbl.filter(singleton_mask).select(["paper_id"])

        writer_clean.write_table(clean_tbl)
        writer_singleton.write_table(singleton_tbl)

        total_clean     += len(clean_tbl)
        total_singleton += len(singleton_tbl)

        del tbl, clean_tbl, singleton_tbl, paper_ids, hashes, is_connected
        gc.collect()

        if (start // WORK_RG_BATCH + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{start + WORK_RG_BATCH}/{n_work_rg} rg]  "
                f"clean: {total_clean:,}  singletons: {total_singleton:,}  ({elapsed:.1f}s)")

    writer_clean.close()
    writer_singleton.close()

    print(f"\nPhase 2 done ({time.time()-t0:.1f}s)")

    # ── Sanity check ──────────────────────────────────────────────────────────────
    total = total_clean + total_singleton
    print(f"\nSanity check:")
    print(f"  work_clean      : {total_clean:,}")
    print(f"  singletons      : {total_singleton:,}")
    print(f"  total           : {total:,}")
    print(f"  original        : 31,787,206")
    print(f"  match           : {total == 31_787_206}")
    # Expected singletons: ~5,309,794
    # Expected clean:      ~26,477,412

def check():
    import pyarrow.parquet as pq
    f = pq.ParquetFile("../data/openalex_mesh/edge_list.parquet")
    print(f"Row groups: {f.metadata.num_row_groups}")
    print(f"Schema: {f.schema}")

clean_edges()

import duckdb
import polars as pl
import numpy as np
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EDGE_PATH      = r".\data\edge_list.parquet"
LINEAGE_PATH   = r".\data\communitylineages\lineages.parquet"
BOOM_PATH      = r".\data\communitylineages\boom_scores.parquet"
GRAPHS_DIR     = Path(r".\data\mesh_processed\community_graphs")
SAMPLES_DIR    = Path(r".\data\mesh_processed\training_samples")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

YEARS      = list(range(1970, 2017))
N_WORKERS  = 10
CHUNK_SIZE = 500

con = duckdb.connect(database=r".\working.duckdb")
con.execute("SET memory_limit = '10GB'")
con.execute("SET threads = 10")
con.execute("SET preserve_insertion_order=false")

print("Tables in DB:", con.execute("SHOW TABLES").fetchall())

# ── boom scores ───────────────────────────────────────────────────────────────
boom_df = pl.read_parquet(BOOM_PATH)

boom_by_lineage = {
    (lid[0] if isinstance(lid, tuple) else lid): grp.sort("year")
    for lid, grp in boom_df.group_by("lineage_id", maintain_order=False)
}

all_lineage_ids = [lid[0] if isinstance(lid, tuple) else lid for lid in boom_by_lineage.keys()]
print(f"Total lineages to process: {len(all_lineage_ids):,}")

# edges_by_lineage is populated per chunk — process_lineage reads from it
edges_by_lineage = {}


# ── per-lineage processing ────────────────────────────────────────────────────
def process_lineage(lid):
    subgraph     = edges_by_lineage.get(lid)
    lineage_boom = boom_by_lineage.get(lid)

    if subgraph is None or lineage_boom is None or len(lineage_boom) == 0:
        return lid, 0

    # write full subgraph parquet — unmasked, all edges for this lineage
    subgraph.write_parquet(GRAPHS_DIR / f"lineage_{lid}.parquet")

    # node registry
    # node_ids: paper_id strings in local index order
    # at training time: node_feats = scibert_lookup(node_ids) -> (N_nodes, 768)
    all_nodes   = list(set(subgraph["src"].to_list() + subgraph["dst"].to_list()))
    node_to_idx = {pid: idx for idx, pid in enumerate(all_nodes)}
    node_ids    = np.array(all_nodes, dtype=object)
    # shape: (N_nodes,)

    # edge arrays — built once, sliced per sample via boolean mask
    src_all  = np.array([node_to_idx[s] for s in subgraph["src"].to_list()], dtype=np.int64)
    dst_all  = np.array([node_to_idx[d] for d in subgraph["dst"].to_list()], dtype=np.int64)
    ts_all   = subgraph["timestamp"].to_numpy().astype(np.int64)
    # msg: dst_in_community as float — 1.0 internal, 0.0 cross-community
    # to add features later: np.concatenate([msg_all, new_feat], axis=1)
    msg_all  = subgraph["dst_in_community"].to_numpy().astype(np.float32).reshape(-1, 1)
    pub_year_all = subgraph["pub_year"].to_numpy().astype(np.int32)
    # shapes: (E_lid,), (E_lid,), (E_lid,), (E_lid, 1), (E_lid,)

    years     = lineage_boom["year"].to_list()
    boom_vals = lineage_boom["boom_score"].to_list()
    n_samples = 0

    for predict_year, boom_label in zip(years, boom_vals):
        # causal mask — TGN only sees edges before predict_year
        mask = pub_year_all < predict_year
        # shape: (E_lid,) boolean

        if mask.sum() == 0:
            continue

        # save as npz — loaded into TemporalData at training time:
        #   data = np.load(path, allow_pickle=True)
        #   TemporalData(
        #       src = torch.from_numpy(data["src"]),  # (E_context,)
        #       dst = torch.from_numpy(data["dst"]),  # (E_context,)
        #       t   = torch.from_numpy(data["t"]),    # (E_context,)
        #       msg = torch.from_numpy(data["msg"]),  # (E_context, 1)
        #       y   = torch.from_numpy(data["y"]),    # (1,)
        #   )
        #   node_feats = scibert_lookup(data["node_ids"])  # (N_nodes, 768)
        np.savez_compressed(
            SAMPLES_DIR / f"lid{lid}_y{predict_year}.npz",
            src          = src_all[mask],                              # (E_context,)
            dst          = dst_all[mask],                              # (E_context,)
            t            = ts_all[mask],                               # (E_context,)
            msg          = msg_all[mask],                              # (E_context, 1)
            y            = np.array([boom_label], dtype=np.float32),  # (1,)
            node_ids     = node_ids,                                   # (N_nodes,)
            lineage_id   = np.array([lid],          dtype=np.int32),
            predict_year = np.array([predict_year], dtype=np.int32),
        )
        n_samples += 1

    return lid, n_samples


# ── chunked extraction + parallel processing ──────────────────────────────────
total_samples = 0
skipped       = 0
done          = 0

print(f"Processing in chunks of {CHUNK_SIZE} with {N_WORKERS} workers...")

for chunk_start in range(0, len(all_lineage_ids), CHUNK_SIZE):
    chunk_ids = all_lineage_ids[chunk_start: chunk_start + CHUNK_SIZE]
    ids_sql   = ",".join(str(i) for i in chunk_ids)

    # extract edges for this chunk only — never loads full 371M edges into RAM
    chunk_edges = con.execute(f"""
        SELECT
            ea_src.lineage_id,
            e.src,
            e.dst,
            e.timestamp,
            e.pub_year,
            CASE WHEN ea_dst.paper_id IS NOT NULL THEN TRUE ELSE FALSE END
                AS dst_in_community
        FROM edges_view e
        JOIN earliest_assign ea_src
          ON ea_src.paper_id   = e.src
         AND e.pub_date >= MAKE_DATE(ea_src.t_assign, 1, 1)
         AND ea_src.lineage_id IN ({ids_sql})
        LEFT JOIN earliest_assign ea_dst
          ON ea_dst.paper_id   = e.dst
         AND ea_dst.lineage_id = ea_src.lineage_id
        ORDER BY ea_src.lineage_id, e.timestamp
    """).pl()

    # populate global dict for process_lineage to read
    edges_by_lineage.clear()
    edges_by_lineage.update({
        (lid[0] if isinstance(lid, tuple) else lid): grp
        for lid, grp in chunk_edges.group_by("lineage_id", maintain_order=True)
    })

    del chunk_edges
    gc.collect()

    # parallel writes for this chunk
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(process_lineage, lid): lid
            for lid in chunk_ids
        }
        for future in as_completed(futures):
            lid, n = future.result()
            done  += 1
            if n == 0:
                skipped += 1
            else:
                total_samples += n

    if chunk_start % 5000 == 0 or chunk_start == 0:
        print(f"  {done:,}/{len(all_lineage_ids):,} done | "
              f"samples: {total_samples:,}")

print(f"\nDone.")
print(f"Total training samples: {total_samples:,}")
print(f"Skipped:                {skipped:,}")

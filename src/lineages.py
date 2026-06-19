import polars as pl
import numpy as np
from collections import defaultdict
import gc
import scipy.sparse as sp
from collections import defaultdict

N_MIN = 10
THETA = 0.30
L_MIN = 2
YEARS = list(range(1970, 2017))

ID_MAP_PATH = "/kaggle/input/notebooks/sujalbajracharya/leiden-community-detection-annually/id_map.parquet"
DATA_DIR = "/kaggle/input/notebooks/sujalbajracharya/leiden-community-detection-annually/leiden_per_year"

print("Loading id_map...")
id_map_df = pl.read_parquet(ID_MAP_PATH)
id_to_int = dict(zip(id_map_df["paper_id"].to_list(), id_map_df["node_idx"].to_list()))
print(f"id_map loaded: {len(id_to_int)} entries")
del id_map_df
gc.collect()

def load_communities(year: int) -> dict[int, frozenset[int]]:
    df = (
        pl.scan_parquet(f"{DATA_DIR}/leiden_{year}.parquet")
        .join(
            pl.scan_parquet(ID_MAP_PATH).select(["paper_id", "node_idx"]),
            on="paper_id",
            how="inner"
        )
        .collect(engine="streaming")
    )
    out = {}
    for cid, group in df.group_by("community_cold"):
        if len(group) >= N_MIN:
            out[cid[0]] = frozenset(group["node_idx"].to_list())
    del df
    return out

print("Loading communities (node_idx encoded)...")
all_comms = {}
for y in YEARS:
    all_comms[y] = load_communities(y)
    gc.collect()
    print(f"  {y}: {len(all_comms[y])}

print("Building cross-year edges (sparse matrix approach)...")
raw_edges = defaultdict(list)
raw_in = defaultdict(list)

def build_sparse_membership(comms: dict) -> tuple[sp.csr_matrix, list, dict]:
    """
    Returns a (n_papers × n_communities) binary membership matrix,
    the ordered list of community_ids, and a paper→row index.
    """
    cids = list(comms.keys())
    cid_to_col = {cid: j for j, cid in enumerate(cids)}

    # collect all unique papers in this year's communities
    all_papers = sorted(set(p for s in comms.values() for p in s))
    paper_to_row = {p: i for i, p in enumerate(all_papers)}

    rows, cols = [], []
    for cid, members in comms.items():
        col = cid_to_col[cid]
        for p in members:
            rows.append(paper_to_row[p])
            cols.append(col)

    n_papers = len(all_papers)
    n_comms = len(cids)
    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_papers, n_comms)
    )
    return M, cids, paper_to_row

for t in YEARS[:-1]:
    comms_t  = all_comms[t]
    comms_t1 = all_comms[t+1]

    if not comms_t or not comms_t1:
        continue

    # Build membership matrices for both years
    # but we need a SHARED paper index across t and t+1
    # so build the union paper set first
    all_papers = sorted(
        set(p for s in comms_t.values() for p in s) |
        set(p for s in comms_t1.values() for p in s)
    )
    paper_to_row = {p: i for i, p in enumerate(all_papers)}
    n_papers = len(all_papers)

    cids_t  = list(comms_t.keys())
    cids_t1 = list(comms_t1.keys())

    def make_matrix(comms, cids, paper_to_row, n_papers):
        rows, cols = [], []
        for j, cid in enumerate(cids):
            for p in comms[cid]:
                rows.append(paper_to_row[p])
                cols.append(j)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_papers, len(cids))
        )

    A = make_matrix(comms_t,  cids_t,  paper_to_row, n_papers)  # n_papers × |C_t|
    B = make_matrix(comms_t1, cids_t1, paper_to_row, n_papers)  # n_papers × |C_{t+1}|

    # Intersection sizes: (|C_t| × |C_{t+1}|) via sparse matmul
    inter = (A.T @ B).toarray()  # shape: (|C_t|, |C_{t+1}|)

    # Sizes
    sizes_t  = np.array([len(comms_t[c])  for c in cids_t],  dtype=np.float32)
    sizes_t1 = np.array([len(comms_t1[c]) for c in cids_t1], dtype=np.float32)

    # Union = size_i + size_j - intersection
    union = sizes_t[:, None] + sizes_t1[None, :] - inter

    # Jaccard matrix — avoid divide by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 0.0)

    # Extract edges >= theta
    rows_idx, cols_idx = np.where(J >= THETA)
    for r, c in zip(rows_idx, cols_idx):
        ci  = cids_t[r]
        cj  = cids_t1[c]
        j   = float(J[r, c])
        raw_edges[(t, ci)].append(((t+1, cj), j))
        raw_in[(t+1, cj)].append(((t, ci), j))

    del A, B, inter, union, J, all_papers, paper_to_row
    gc.collect()
    print(f"  {t}→{t+1} done")

print("Edges built.")

def is_one_to_one_forward(node):
    out = raw_edges.get(node, [])
    if len(out) != 1:
        return None
    target, j = out[0]
    if len(raw_in.get(target, [])) != 1:
        return None
    return target

def is_one_to_one_backward(node):
    inc = raw_in.get(node, [])
    if len(inc) != 1:
        return None
    source, j = inc[0]
    if len(raw_edges.get(source, [])) != 1:
        return None
    return source

all_nodes = [(y, cid) for y in YEARS for cid in all_comms[y]]

lineage_of = {}
lineage_rows = []
lineage_id_counter = 0

heads = [n for n in all_nodes if is_one_to_one_backward(n) is None]

for head in heads:
    lid = lineage_id_counter
    lineage_id_counter += 1
    cur = head
    while True:
        lineage_of[cur] = lid
        lineage_rows.append({"lineage_id": lid, "year": cur[0], "community_id": cur[1]})
        nxt = is_one_to_one_forward(cur)
        if nxt is None:
            break
        cur = nxt

lineage_df = pl.DataFrame(lineage_rows)
print(f"Total lineage rows: {len(lineage_rows)}, total lineages: {lineage_id_counter}")

events = []
for node in all_nodes:
    out_edges = raw_edges.get(node, [])
    in_edges = raw_in.get(node, [])
    src_lid = lineage_of.get(node)

    if len(out_edges) == 0 and node[0] != YEARS[-1]:
        events.append({"type": "death", "year": node[0], "lineage_id": src_lid,
                        "community_id": node[1], "target_lineage_id": None, "J": None})
    elif len(out_edges) > 1:
        for target, j in out_edges:
            tgt_lid = lineage_of.get(target)
            events.append({"type": "split", "year": node[0], "lineage_id": src_lid,
                            "community_id": node[1], "target_lineage_id": tgt_lid, "J": j})

    if len(in_edges) == 0 and node[0] != YEARS[0]:
        events.append({"type": "birth", "year": node[0], "lineage_id": src_lid,
                        "community_id": node[1], "target_lineage_id": None, "J": None})
    elif len(in_edges) > 1:
        for source, j in in_edges:
            src_lid2 = lineage_of.get(source)
            events.append({"type": "merge", "year": node[0], "lineage_id": src_lid2,
                            "community_id": source[1], "target_lineage_id": src_lid, "J": j})

# events_df = pl.DataFrame(events)
events_df = pl.DataFrame(events, schema={
    "type": pl.Utf8,
    "year": pl.Int32,
    "lineage_id": pl.Int64,
    "community_id": pl.Int64,
    "target_lineage_id": pl.Int64,
    "J": pl.Float64,
})

lineage_lengths = lineage_df.group_by("lineage_id").agg(pl.len().alias("length"))

print(lineage_df.group_by("lineage_id").agg(
    pl.col("year").min().alias("start"),
    pl.col("year").max().alias("end"),
    pl.len().alias("length"),
).describe())

n_total = lineage_lengths.height
n_above_lmin = lineage_lengths.filter(pl.col("length") >= L_MIN).height
print(f"\nTotal lineages: {n_total} | with length >= {L_MIN}: {n_above_lmin}")
print(events_df["type"].value_counts())

lineage_df.write_parquet("lineages.parquet")
events_df.write_parquet("lineage_events.parquet")
print("Saved.")
print(events_df["type"].value_counts())

import polars as pl

# Get start/end/length per lineage
lineage_summary = lineage_df.group_by("lineage_id").agg(
    pl.col("year").min().alias("start"),
    pl.col("year").max().alias("end"),
    pl.len().alias("length"),
)

# 1. Overall: where do length>=5 lineages start and end?
long_lineages = lineage_summary.filter(pl.col("length") >= L_MIN)

print("Start-year distribution of length>=2 lineages:")
print(long_lineages.group_by("start").agg(pl.len().alias("n")).sort("start"))

print("\nEnd-year distribution of length>=2 lineages:")
print(long_lineages.group_by("end").agg(pl.len().alias("n")).sort("end"))

# 2. Specifically: how many length>=5 lineages SPAN across the 1996-2005 "dead zone"?
spans_deadzone = long_lineages.filter(
    (pl.col("start") < 1996) & (pl.col("end") > 2005)
)
print(f"\nLength>=5 lineages spanning across 1996-2005: {spans_deadzone.height}")

# 3. Mean length by decade-of-start, all lineages (not just >=5)
print("\nMean length by start-decade (all lineages):")
print(
    lineage_summary
    .with_columns((pl.col("start") // 10 * 10).alias("decade"))
    .group_by("decade")
    .agg(pl.len().alias("n_lineages"), pl.col("length").mean().alias("mean_length"))
    .sort("decade")
)

# 4. For the length>=5 cohort, what's the death-year distribution relative to the
#    530->50 community collapse window (roughly 1996-2009 per your community counts)?
print("\nLength>=2 lineages: how many END in the 1996-2009 collapse window?")
ended_in_collapse = long_lineages.filter((pl.col("end") >= 1996) & (pl.col("end") <= 2009))
print(f"{ended_in_collapse.height} / {long_lineages.height}")

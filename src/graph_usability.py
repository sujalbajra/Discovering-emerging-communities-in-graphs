# ============================================================
#  GRAPH USABILITY AUDIT  —  OpenAlex MeSH Citation Network
#  v5: fixes Phase 3 OOM — 637M edges never fully in RAM
#
#  Root cause of crash:
#    pairs struct (637M×16B=10GB) + src_idx + dst_idx + all_hashes
#    all live simultaneously → ~15GB → kernel OOM kill
#
#  Fix strategy:
#    1. Sort+dedup pairs in-place (already done, good)
#    2. Write deduped (src_hash, dst_hash) to a memory-mapped
#       temp file immediately after dedup → free pairs from RAM
#    3. Build remap table from node_hashes only (not edges)
#       → extends lazily for unknown src via searchsorted
#    4. Stream src_idx/dst_idx in chunks from mmap for:
#       - degree bincount
#       - CSR construction (via incremental COO → CSR)
#    5. WCC on CSR with directed=False (no mat+mat.T copy)
#
#  Peak RAM budget:
#    pairs during sort:     ~10 GB  (unavoidable, already present)
#    After free(pairs):     ~0.5 GB (mmap on disk, not RAM)
#    src_idx chunk (100M):  ~0.4 GB
#    CSR data arrays:       ~3-4 GB total
#    Total peak:            ~12 GB  → fits in 16 GB
# ============================================================

import gc
import os
import sys
import time
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_wcc

# ─────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────
WORK_PATH    = "/kaggle/input/datasets/sujalbajracharya/openalex-mesh-only/mesh_data/work.parquet"
EDGE_PATH    = "/kaggle/input/datasets/sujalbajracharya/openalex-mesh-only/edge_list.parquet"
MESH_PATH    = "/kaggle/input/datasets/sujalbajracharya/openalex-mesh-only/mesh_data/mesh_data.parquet"
WORK_SIZE_GB = 7.73
EDGE_SIZE_GB = 3.86
MESH_SIZE_GB = 1.65

WORK_RG_BATCH = 8
EDGE_RG_BATCH = 20

# Chunk size for streaming remap+degree+CSR build
# 100M × 8B × 2 arrays = 1.6 GB peak per chunk
REMAP_CHUNK = 50_000_000

# ─────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────
try:
    import xxhash
    def hash_arr(lst):
        return np.array([xxhash.xxh64_intdigest(s) for s in lst], dtype=np.uint64)
    print("[Setup]  xxhash available — using xxh64")
except ImportError:
    import hashlib
    def hash_arr(lst):
        return np.array([
            int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'little')
            for s in lst], dtype=np.uint64)
    print("[Setup]  xxhash not found — using sha256 fallback")

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"None of {candidates} found in {cols}")

def read_rg_batch(pf, rg_list, cols):
    return pa.concat_tables([pf.read_row_group(rg, columns=cols) for rg in rg_list])

def batched_rg_ranges(n, batch):
    for s in range(0, n, batch):
        yield list(range(s, min(s + batch, n)))

def mem_gb():
    """Current process RSS in GB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return 0.0

print("=" * 65)
print("  GRAPH USABILITY AUDIT — OpenAlex MeSH Citation Network")
print("  v5: stream-to-disk after dedup, chunked remap+CSR build")
print("=" * 65)

# ─────────────────────────────────────────────────────────
#  SCHEMA
# ─────────────────────────────────────────────────────────
work_cols = pq.read_schema(WORK_PATH).names
edge_cols = pq.read_schema(EDGE_PATH).names
WORK_ID   = pick_col(work_cols, ["paper_id", "id", "work_id", "openalex_id"])
EDGE_SRC  = pick_col(edge_cols, ["paper_id", "citing_id", "src", "source"])
EDGE_DST  = pick_col(edge_cols, ["cited_id",  "reference_id", "dst", "target"])

pf_work   = pq.ParquetFile(WORK_PATH)
pf_edge   = pq.ParquetFile(EDGE_PATH)
n_work_rg = pf_work.metadata.num_row_groups
n_edge_rg = pf_edge.metadata.num_row_groups

print(f"\n[Schema]  node_id={WORK_ID}  src={EDGE_SRC}  dst={EDGE_DST}")
print(f"  work rg={n_work_rg}   edge rg={n_edge_rg}")

# ─────────────────────────────────────────────────────────
#  PHASE 1 — NODE HASH SET  (unchanged from v4)
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 1] Building node hash set …")
t0 = time.time()

node_hashes_list = []
for rg_batch in batched_rg_ranges(n_work_rg, WORK_RG_BATCH):
    tbl = read_rg_batch(pf_work, rg_batch, [WORK_ID])
    ids = pc.unique(pc.drop_null(tbl.column(WORK_ID))).to_pylist()
    node_hashes_list.append(hash_arr(ids))
    del tbl, ids

node_hashes_arr = np.unique(np.concatenate(node_hashes_list))
del node_hashes_list
total_nodes   = len(node_hashes_arr)
node_hash_set = set(node_hashes_arr.tolist())
print(f"  Unique nodes : {total_nodes:,}   ({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# ─────────────────────────────────────────────────────────
#  PHASE 2 — EDGE SCAN  (unchanged — already works fine)
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 2] Edge scan …")
t0 = time.time()

total_edges        = 0
null_count         = 0
self_cite_count    = 0
invalid_edge_count = 0
src_chunks         = []
dst_chunks         = []

for rg_batch in batched_rg_ranges(n_edge_rg, EDGE_RG_BATCH):
    tbl       = read_rg_batch(pf_edge, rg_batch, [EDGE_SRC, EDGE_DST])
    src_col   = tbl.column(EDGE_SRC)
    dst_col   = tbl.column(EDGE_DST)
    batch_len = len(src_col)
    total_edges += batch_len

    valid_mask = pc.and_(pc.is_valid(src_col), pc.is_valid(dst_col))
    n_nulls    = batch_len - int(pc.sum(valid_mask).as_py())
    null_count += n_nulls
    if n_nulls > 0:
        tbl     = tbl.filter(valid_mask)
        src_col = tbl.column(EDGE_SRC)
        dst_col = tbl.column(EDGE_DST)

    src_list = src_col.to_pylist()
    dst_list = dst_col.to_pylist()
    del tbl, src_col, dst_col, valid_mask

    batch_src = []
    batch_dst = []
    for s, d in zip(src_list, dst_list):
        if s == d:
            self_cite_count += 1
        dh = xxhash.xxh64_intdigest(d) if 'xxhash' in sys.modules else hash_arr([d])[0]
        if dh not in node_hash_set:
            invalid_edge_count += 1
            continue
        sh = xxhash.xxh64_intdigest(s) if 'xxhash' in sys.modules else hash_arr([s])[0]
        batch_src.append(sh)
        batch_dst.append(dh)

    if batch_src:
        src_chunks.append(np.array(batch_src, dtype=np.uint64))
        dst_chunks.append(np.array(batch_dst, dtype=np.uint64))
    del src_list, dst_list, batch_src, batch_dst

    done_rg = rg_batch[-1] + 1
    print(f"  rg {done_rg}/{n_edge_rg} | raw:{total_edges:,} inv:{invalid_edge_count:,} "
          f"t:{time.time()-t0:.0f}s", end="\r")

print()
invalid_edge_count += null_count
edge_coverage  = (total_edges - invalid_edge_count) / total_edges * 100 if total_edges else 0.0
self_cite_rate = self_cite_count / total_edges * 100 if total_edges else 0.0

print(f"  Raw:{total_edges:,}  Nulls:{null_count:,}  Invalid:{invalid_edge_count-null_count:,}  "
      f"Self-cite:{self_cite_count:,}  Coverage:{edge_coverage:.2f}%  RSS:{mem_gb():.1f}GB")

if edge_coverage < 90.0:
    print(f"\n{'!'*65}\n  STOP: DATASET MISMATCH — coverage {edge_coverage:.2f}% < 90%\n{'!'*65}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
#  PHASE 3a — SORT+DEDUP  (same as before, in-place)
#  After this we IMMEDIATELY write to disk and free pairs.
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 3a] Sort + dedup …   RSS:{mem_gb():.1f}GB")
t0 = time.time()

src_all = np.concatenate(src_chunks)
dst_all = np.concatenate(dst_chunks)
del src_chunks, dst_chunks
gc.collect()

raw_valid = len(src_all)

pairs     = np.empty(raw_valid, dtype=[('s', np.uint64), ('d', np.uint64)])
pairs['s'] = src_all
pairs['d'] = dst_all
del src_all, dst_all
gc.collect()

pairs.sort()

keep     = np.empty(raw_valid, dtype=bool)
keep[0]  = True
keep[1:] = (pairs['s'][1:] != pairs['s'][:-1]) | (pairs['d'][1:] != pairs['d'][:-1])
pairs    = pairs[keep]
del keep
gc.collect()

dup_count   = raw_valid - len(pairs)
valid_edges = len(pairs)
dup_rate    = dup_count / total_edges * 100 if total_edges else 0.0
print(f"  Dedup: {raw_valid:,} → {valid_edges:,}  removed:{dup_count:,}  "
      f"({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# ─────────────────────────────────────────────────────────
#  PHASE 3b — WRITE DEDUPED EDGES TO DISK, FREE pairs
#
#  Write as a flat binary file: [src_uint64 | dst_uint64]
#  interleaved, so a memmap gives us back the pairs cheaply.
#  Disk usage: 637M × 16 bytes ≈ 10 GB  (Kaggle has 20 GB disk)
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 3b] Writing deduped edges to disk …")
t0 = time.time()

edge_tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir="/tmp")
edge_tmp_path = edge_tmp.name
edge_tmp.close()

# Write interleaved: s0 d0 s1 d1 ...
flat = np.empty(valid_edges * 2, dtype=np.uint64)
flat[0::2] = pairs['s']
flat[1::2] = pairs['d']
del pairs
gc.collect()

flat.tofile(edge_tmp_path)
del flat
gc.collect()
print(f"  Written {valid_edges*16/1e9:.2f} GB to {edge_tmp_path}  "
      f"({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# ─────────────────────────────────────────────────────────
#  PHASE 3c — BUILD REMAP TABLE
#
#  Remap table = sorted unique hashes across ALL edge endpoints
#  + known nodes.  But we can't load all edges to get unique
#  hashes.  Instead: since edge scan already filtered dst to
#  known nodes, dst hashes ⊆ node_hashes_arr.
#  src hashes may include unknown citing papers.
#
#  Strategy: stream the mmap'd edge file to collect unique
#  src hashes in a set, merge with node_hashes_arr, sort.
#  Memory: set of unique src hashes ≈ up to 31M × 8B ≈ 248 MB
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 3c] Building remap table …")
t0 = time.time()

mmap_edges = np.memmap(edge_tmp_path, dtype=np.uint64, mode='r')
# View as (N, 2): col0=src, col1=dst
mmap_pairs = mmap_edges.reshape(-1, 2)

# Collect unique src hashes by streaming chunks
src_hash_set = set()
for start in range(0, valid_edges, REMAP_CHUNK):
    chunk = mmap_pairs[start : start + REMAP_CHUNK, 0]
    src_hash_set.update(chunk.tolist())

# All dst hashes are in node_hashes_arr by construction
# Extra src hashes not in node set
extra_src = np.array(
    [h for h in src_hash_set if h not in node_hash_set],
    dtype=np.uint64
)
del src_hash_set
gc.collect()

all_hashes = np.unique(np.concatenate([node_hashes_arr, extra_src]))
del extra_src, node_hashes_arr
gc.collect()

n_vocab = len(all_hashes)
print(f"  Vocab size: {n_vocab:,}  ({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# ─────────────────────────────────────────────────────────
#  PHASE 4 — DEGREE DISTRIBUTION  (chunked, from mmap)
#
#  Stream edge file in REMAP_CHUNK slices.
#  Remap via searchsorted (O(log V) per element, vectorised).
#  Accumulate degree counts in two int32 arrays of size n_vocab.
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 4] Degree distribution (chunked from mmap) …")
t0 = time.time()

out_counts = np.zeros(n_vocab, dtype=np.int32)
in_counts  = np.zeros(n_vocab, dtype=np.int32)

for start in range(0, valid_edges, REMAP_CHUNK):
    chunk   = mmap_pairs[start : start + REMAP_CHUNK]
    s_chunk = np.searchsorted(all_hashes, chunk[:, 0]).astype(np.int32)
    d_chunk = np.searchsorted(all_hashes, chunk[:, 1]).astype(np.int32)
    np.add.at(out_counts, s_chunk, 1)
    np.add.at(in_counts,  d_chunk, 1)
    del s_chunk, d_chunk, chunk

out_nz   = out_counts[out_counts > 0]
in_nz    = in_counts[in_counts  > 0]
max_out  = int(out_nz.max())    if len(out_nz) else 0
max_in   = int(in_nz.max())     if len(in_nz)  else 0
mean_out = float(out_nz.mean()) if len(out_nz) else 0.0
mean_in  = float(in_nz.mean())  if len(in_nz)  else 0.0
p99_out  = float(np.percentile(out_nz, 99)) if len(out_nz) else 0.0
p99_in   = float(np.percentile(in_nz,  99)) if len(in_nz)  else 0.0
heavy_out = (p99_out / mean_out > 10) if mean_out > 0 else False
heavy_in  = (p99_in  / mean_in  > 10) if mean_in  > 0 else False

# Singletons: known nodes with both degrees == 0
node_idx_in_vocab = np.searchsorted(all_hashes, np.sort(np.array(list(node_hash_set), dtype=np.uint64)))
# Guard against hash collisions / off-by-one
valid_lookup = node_idx_in_vocab < n_vocab
node_idx_in_vocab = node_idx_in_vocab[valid_lookup]
singleton_count = int(
    ((out_counts[node_idx_in_vocab] == 0) & (in_counts[node_idx_in_vocab] == 0)).sum()
)
del out_nz, in_nz, out_counts, in_counts, node_idx_in_vocab, valid_lookup, node_hash_set
gc.collect()

print(f"  Out — max:{max_out:,}  mean:{mean_out:.2f}  p99:{p99_out:.1f}  heavy:{heavy_out}")
print(f"  In  — max:{max_in:,}   mean:{mean_in:.2f}  p99:{p99_in:.1f}  heavy:{heavy_in}")
print(f"  Singletons: {singleton_count:,}   ({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# ─────────────────────────────────────────────────────────
#  PHASE 5 — CSR BUILD + WCC  (chunked COO → CSR)
#
#  Stream edges from mmap in chunks, remap to int32 indices,
#  build COO arrays incrementally, then convert to CSR once.
#  Memory: COO row+col arrays = 2 × valid_edges × 4B = ~5 GB
#  This is unavoidable for SciPy CSR — it needs all indices.
#  We build them in int32 to halve cost vs int64.
# ─────────────────────────────────────────────────────────
print(f"\n[Phase 5] Building COO arrays for CSR …")
t0 = time.time()

# Pre-allocate full int32 COO arrays (5 GB total — fits with ~6 GB headroom)
coo_row = np.empty(valid_edges, dtype=np.int32)
coo_col = np.empty(valid_edges, dtype=np.int32)

for start in range(0, valid_edges, REMAP_CHUNK):
    end     = min(start + REMAP_CHUNK, valid_edges)
    chunk   = mmap_pairs[start:end]
    coo_row[start:end] = np.searchsorted(all_hashes, chunk[:, 0]).astype(np.int32)
    coo_col[start:end] = np.searchsorted(all_hashes, chunk[:, 1]).astype(np.int32)
    del chunk

del mmap_edges, mmap_pairs, all_hashes
gc.collect()

# Clean up temp file
try:
    os.unlink(edge_tmp_path)
except Exception:
    pass

print(f"  COO built — {valid_edges:,} edges   ({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

data = np.ones(valid_edges, dtype=np.bool_)
mat  = csr_matrix((data, (coo_row, coo_col)), shape=(n_vocab, n_vocab))
del data, coo_row, coo_col
gc.collect()

print(f"  CSR shape:{mat.shape} nnz:{mat.nnz:,}   ({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# directed=False: scipy handles undirected WCC without mat+mat.T
num_components, labels = sp_wcc(mat, directed=False, connection="weak")
del mat
gc.collect()

comp_sizes   = np.bincount(labels)
largest_size = int(comp_sizes.max())
del labels, comp_sizes
largest_pct  = largest_size / total_nodes * 100
print(f"  Components:{num_components:,}  Largest:{largest_size:,} ({largest_pct:.2f}%)  "
      f"({time.time()-t0:.1f}s)  RSS:{mem_gb():.1f}GB")

# ─────────────────────────────────────────────────────────
#  FINAL REPORT
# ─────────────────────────────────────────────────────────
audit_warnings = []
if singleton_count / total_nodes > 0.20:
    audit_warnings.append(f"High singleton ratio: {singleton_count/total_nodes*100:.1f}%")
if largest_pct < 50.0:
    audit_warnings.append(f"Largest component only {largest_pct:.1f}% of nodes")
if dup_rate > 5.0:
    audit_warnings.append(f"High duplicate edge rate: {dup_rate:.2f}%")
if self_cite_rate > 5.0:
    audit_warnings.append(f"High self-citation rate: {self_cite_rate:.2f}%")
if heavy_in or heavy_out:
    audit_warnings.append("Heavy-tailed degree distribution — expected for citation networks")

decision = "STOP AND FIX PIPELINE" if edge_coverage < 90.0 else \
           "PROCEED WITH WARNINGS"  if audit_warnings          else \
           "PROCEED"

print("\n")
print("=" * 65)
print("  GRAPH AUDIT SUMMARY")
print("=" * 65)
print(f"\n  PATHS")
print(f"    work  : {WORK_PATH}")
print(f"    edges : {EDGE_PATH}")
print(f"    mesh  : {MESH_PATH}")
print(f"\n  SIZES  work:{WORK_SIZE_GB}GB  edges:{EDGE_SIZE_GB}GB  "
      f"mesh:{MESH_SIZE_GB}GB  total:{WORK_SIZE_GB+EDGE_SIZE_GB+MESH_SIZE_GB:.2f}GB")
print(f"\n  BACKEND : CPU — PyArrow + NumPy + SciPy + disk-spill [v5]")
print(f"\n  GRAPH STRUCTURE")
print(f"    Total nodes            : {total_nodes:,}")
print(f"    Total edges (raw)      : {total_edges:,}")
print(f"    Invalid edges removed  : {invalid_edge_count:,}")
print(f"    Duplicate edges removed: {dup_count:,}")
print(f"    Valid unique edges     : {valid_edges:,}")
print(f"    Edge coverage          : {edge_coverage:.2f}%")
print(f"\n  CONNECTIVITY")
print(f"    Weakly connected components : {num_components:,}")
print(f"    Largest component           : {largest_size:,} ({largest_pct:.2f}%)")
print(f"    Singleton nodes (deg=0)     : {singleton_count:,} ({singleton_count/total_nodes*100:.2f}%)")
print(f"\n  DEGREE DISTRIBUTION")
print(f"    In-degree  — max:{max_in:,}  mean:{mean_in:.2f}  heavy-tail:{heavy_in}")
print(f"    Out-degree — max:{max_out:,}  mean:{mean_out:.2f}  heavy-tail:{heavy_out}")
print(f"\n  QUALITY METRICS")
print(f"    Self-citation rate  : {self_cite_rate:.4f}%")
print(f"    Duplicate edge rate : {dup_rate:.4f}%")

if audit_warnings:
    print(f"\n  WARNINGS ({len(audit_warnings)})")
    for w in audit_warnings:
        print(f"    ⚠  {w}")

print(f"\n{'=' * 65}")
print(f"  FINAL DECISION:  {decision}")
print(f"{'=' * 65}\n")

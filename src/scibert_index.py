import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

SCIBERT_DIR = Path(r".\scibert_embeddings")
HDF5_PATH   = Path(r".\sci_index\scibert_index.h5")

# ── Pass 1: collect all paper IDs across all embedding files ──────────────────
# We need a global sorted list to assign integer indices
# paper_id (string) -> integer row index in HDF5 dataset

print("Pass 1: scanning all embedding files for paper IDs...")
all_ids = []

npz_files = sorted(SCIBERT_DIR.glob("*.npz"))
print(f"Found {len(npz_files)} embedding files")

for f in tqdm(npz_files):
    d = np.load(f, allow_pickle=True)
    all_ids.append(d["ids"])  # shape: (50000,) strings

all_ids = np.concatenate(all_ids)  # shape: (N_total,)
print(f"Total papers in SciBERT index: {len(all_ids):,}")

# Build paper_id -> row_index mapping
# This dict lives in RAM but it's just strings->ints, not embeddings
# ~10M entries * ~60 bytes per entry ≈ 600MB — acceptable
print("Building paper_id -> row_index mapping...")
id_to_row = {pid: i for i, pid in enumerate(all_ids)}
print(f"Index map built. Size: {len(id_to_row):,}")

# ── Pass 2: write embeddings into HDF5 in order ───────────────────────────────
# Layout: single dataset "embeddings" of shape (N_total, 768)
# Row i = embedding for paper all_ids[i]
# Chunked + gzip compressed for fast row-wise access

print(f"\nPass 2: writing HDF5 to {HDF5_PATH}...")

N_total = len(all_ids)
CHUNK_ROWS = 1000  # HDF5 chunk size — controls random access granularity

with h5py.File(HDF5_PATH, "w") as hf:
    # Fix: cast to object dtype so h5py vlen string writer accepts it
    all_ids_obj = all_ids.astype(object)  # dtype('<U11') -> dtype('O')
    # shape: (N_total,) object array of Python strings

    dt = h5py.string_dtype(encoding='utf-8')  # modern h5py API, replaces special_dtype
    hf.create_dataset(
        "paper_ids",
        data=all_ids_obj,   # (N_total,) object strings
        dtype=dt,
        chunks=(CHUNK_ROWS,),
    )

    emb_ds = hf.create_dataset(
        "embeddings",
        shape=(N_total, 768),
        dtype=np.float32,
        chunks=(CHUNK_ROWS, 768),
        compression="gzip",
        compression_opts=4,
    )
    # shape: (N_total, 768)

    row_offset = 0
    for f in tqdm(npz_files):
        d = np.load(f, allow_pickle=True)
        emb = d["embeddings"]   # (50000, 768)
        n   = len(emb)
        emb_ds[row_offset: row_offset + n] = emb
        row_offset += n

print(f"HDF5 written. Total rows: {row_offset:,}")
print(f"File size: {HDF5_PATH.stat().st_size / 1e9:.2f} GB")

# Save the id_to_row dict as numpy for fast reloading
# Faster than rebuilding from HDF5 every run
np.save(
    r".\sci_index\scibert_id_to_row.npy",
    id_to_row,
    allow_pickle=True,
)
print("id_to_row mapping saved.")

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

id_to_row = np.load(r'.\sci_index\scibert_id_to_row.npy', allow_pickle=True).item()

samples_dir = Path(r'.\graphs\training_samples')
files = list(samples_dir.glob('*.npz'))

missing_any = 0
total_missing_nodes = 0
total_nodes = 0

for f in tqdm(files[:500]):  # check 500 samples
# for f in tqdm(files):  # check 500 samples
    d = np.load(f, allow_pickle=True)
    node_ids = d['node_ids']
    missing = [pid for pid in node_ids if pid not in id_to_row]
    total_nodes += len(node_ids)
    total_missing_nodes += len(missing)
    if missing:
        missing_any += 1

print(f"Samples with at least one missing node: {missing_any}/{len(files)}")
print(f"Total missing nodes: {total_missing_nodes}/{total_nodes} ({100*total_missing_nodes/total_nodes:.2f}%)")

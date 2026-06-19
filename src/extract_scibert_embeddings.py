import os
import gc
import re
import numpy as np
import duckdb
import torch
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 1. ENVIRONMENT & HARDWARE INITIALIZATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using computational backend: {device}")

print("Loading SciBERT vocabulary and neural network weights...")
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()  # Freeze layers for inference optimization

# Configure paths and parameters
INPUT_DATASET = "openalex_abstracts_only.parquet"
OUTPUT_DIR = "./scibert_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

batch_size = 16       # VRAM target chunk size
chunk_size = 50000     # System RAM target extraction size
END_ROW = 22127580     # Hard boundary of dataset size

# ==========================================
# 2. DYNAMIC BOOKMARK / RESUME SCANNER
# ==========================================
print("\nScanning storage directory for previous progress...")
detected_max_row = 0

if os.path.exists(OUTPUT_DIR):
    for filename in os.listdir(OUTPUT_DIR):
        # Scan for old format: embeddings_part_300.npz (each contains 50k items)
        part_match = re.match(r"embeddings_part_(\d+)\.npz", filename)
        if part_match:
            chunk_idx = int(part_match.group(1))
            next_row = (chunk_idx + 1) * 50000
            if next_row > detected_max_row:
                detected_max_row = next_row

        # Scan for streaming format: embeddings_row_15000000.npz
        row_match = re.match(r"embeddings_row_(\d+)\.npz", filename)
        if row_match:
            row_offset = int(row_match.group(1))
            next_row = row_offset + chunk_size
            if next_row > detected_max_row:
                detected_max_row = next_row

START_ROW = detected_max_row
TOTAL_ROWS_TO_PROCESS = END_ROW - START_ROW

if START_ROW >= END_ROW:
    print(f"Database already fully processed up to target end row ({END_ROW:,})!")
    import sys; sys.exit()

print(f"--> SUCCESS: Auto-resume targeted. Resuming pipeline from Row: {START_ROW:,}")
print(f"--> Remaining workload target: {TOTAL_ROWS_TO_PROCESS:,} rows.")

# ==========================================
# 3. DATABASE STREAM ROUTING (DUCKDB)
# ==========================================
ctx = duckdb.connect()
ctx.execute("SET threads=4")                       # Prevents thread-memory spikes
ctx.execute("SET preserve_insertion_order=false")  # Drops internal position mapping tracking

# Define master data window selection query
master_query = f"""
    SELECT openalex_id, LOWER(TRIM(clean_abstract)) as processed_text
    FROM '{INPUT_DATASET}'
    WHERE clean_abstract IS NOT NULL
    LIMIT {TOTAL_ROWS_TO_PROCESS} OFFSET {START_ROW}
"""

print("Spooling master dataset file stream cursor...")
cursor = ctx.execute(master_query)
current_offset = START_ROW

# ==========================================
# 4. CORE PROCESSING & PIPELINE LOOP
# ==========================================
while current_offset < END_ROW:
    print(f"--> STREAMING CHUNK (Current Row Position: {current_offset:,})...")

    # Extract blocks directly into lightweight native tuples
    batch_data = cursor.fetchmany(chunk_size)
    if not batch_data:
        print("Data stream naturally exhausted.")
        break

    chunk_ids = [row[0] for row in batch_data]
    chunk_texts = [row[1] for row in batch_data]

    compiled_embeddings = []

    # Process small sub-batches on the GPU
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Inference mode prevents tracking history, drastically saving VRAM
        with torch.inference_mode():
            outputs = model(**inputs)
            # Isolate the [CLS] token (index 0) vector
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            compiled_embeddings.append(cls_embeddings)

    # Merge matrix segments together
    chunk_vectors = np.vstack(compiled_embeddings)

    # Commit compressed block to disk tagged by its unique row coordinate
    output_file = os.path.join(OUTPUT_DIR, f"embeddings_row_{current_offset}.npz")
    np.savez_compressed(output_file, ids=chunk_ids, embeddings=chunk_vectors)

    # Push stream offset tracker forward based on rows actually yielded
    current_offset += len(batch_data)

    # Force RAM and VRAM garbage collection tracking resets
    del batch_data, chunk_ids, chunk_texts, chunk_vectors, compiled_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll remaining target rows successfully completed and secured!")

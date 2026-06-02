import duckdb

con = duckdb.connect()

# ── 2. Explode list column + count papers per tag ─────────────────────────────
# UNNEST expands List(String) → one row per element
# DuckDB streams this off disk — never loads full file into RAM
papers_per_tag = con.sql("""
    SELECT
        descriptor_id,
        COUNT(DISTINCT paper_id) AS paper_count
    FROM (
        SELECT
            paper_id,
            UNNEST(descriptor_ids) AS descriptor_id
        FROM '../data/openalex_mesh/mesh_data.parquet'
    )
    GROUP BY descriptor_id
    ORDER BY paper_count DESC
""").df()   # .df() pulls result into a small pandas DataFrame (only tag-level aggregates)

# ── 3. Join with descriptor metadata ─────────────────────────────────────────
descs = con.sql("SELECT * FROM '../data/openalex_mesh/descriptor_data.parquet'").df()

tag_stats = papers_per_tag.merge(descs, on="descriptor_id", how="left")
tag_stats = tag_stats.sort_values("paper_count", ascending=False).reset_index(drop=True)
multi = tag_stats[tag_stats["paper_count"] > 1]

# ── 4. Summary stats ──────────────────────────────────────────────────────────
print(f"Unique tags with >= 1 paper : {len(tag_stats):,}")
print(f"\nTags with > 1 paper: {len(multi):,}")
print(f"Mean papers/tag             : {tag_stats['paper_count'].mean():,.1f}")
print(f"Median papers/tag           : {tag_stats['paper_count'].median():,.1f}")
print(f"Max                         : {tag_stats['paper_count'].max():,}")
print(f"Min                         : {tag_stats['paper_count'].min():,}")

tag_stats.to_csv("../data/mesh_processed/mesh_tag_stats.csv", index=False)

# ── 5. Power-law bucket check ─────────────────────────────────────────────────
import pandas as pd

bins   = [1, 10, 100, 1_000, 10_000, 100_000, 10_000_000]
labels = ["1-9", "10-99", "100-999", "1k-9k", "10k-99k", "100k+"]

tag_stats["bucket"] = pd.cut(
    tag_stats["paper_count"], bins=bins, labels=labels, right=False
)
print("\nBucket distribution:")
print(
    tag_stats.groupby("bucket", observed=True)["paper_count"]
    .count()
    .rename("tag_count")
    .to_string()
)

# ── 6. Major-topic fraction ───────────────────────────────────────────────────
major_stats = con.sql("""
    SELECT
        ROUND(
            100.0 * SUM(CASE WHEN is_major = TRUE THEN 1 ELSE 0 END) / COUNT(*),
            2
        ) AS major_topic_pct
    FROM (
        SELECT
            UNNEST(descriptor_ids)  AS descriptor_id,
            UNNEST(is_major_topics) AS is_major
        FROM '../data/openalex_mesh/mesh_data.parquet'
    )
""").df()
print("\nMajor-topic assignments (%):")
print(major_stats)

# ── 7. Rare tag count ─────────────────────────────────────────────────────────
rare = tag_stats[tag_stats["paper_count"] < 10]
print(f"\nRare tags (< 10 papers): {len(rare):,}")

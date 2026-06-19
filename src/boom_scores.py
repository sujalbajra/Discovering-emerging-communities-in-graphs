import duckdb
import polars as pl
from pathlib import Path

CITATION_PATH  = "/kaggle/input/notebooks/rnewr013/openalex-mesh-22m/citation_count.parquet"
LINEAGE_PATH   = "/kaggle/input/datasets/sujalbajracharya/community-lineages/lineages.parquet"
LEIDEN_DIR     = "/kaggle/input/notebooks/sujalbajracharya/leiden-community-detection-annually/leiden_per_year"
BOOM_OUT       = "boom_scores.parquet"
EDGE_PATH      = "/kaggle/input/notebooks/rnewr013/openalex-mesh-22m/edge_list.parquet"
YEARS          = list(range(1970, 2017))

con = duckdb.connect(database="working.duckdb")

# ── step 1: citation counts per paper per year from edge list ────────────────
# cited_id receives one citation for each edge pointing to it
# citation year = year of publication_date of the CITING paper (src)
# so we group by cited_id + year(publication_date) and count

con.execute("DROP TABLE IF EXISTS yearly_citations")
con.execute(f"""
    CREATE TABLE yearly_citations AS
    SELECT
        cited_id                                    AS paper_id,
        YEAR(publication_date::DATE)                AS citation_year,
        COUNT(*)                                    AS yearly_citations
    FROM read_parquet('{EDGE_PATH}')
    GROUP BY cited_id, YEAR(publication_date::DATE)
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_yc ON yearly_citations (paper_id, citation_year)")
print("yearly_citations built.")

# ── step 2: cumulative citations per paper per year ──────────────────────────
con.execute("DROP TABLE IF EXISTS cum_cit")
con.execute("""
    CREATE TABLE cum_cit AS
    SELECT
        paper_id,
        citation_year,
        SUM(yearly_citations) OVER (
            PARTITION BY paper_id
            ORDER BY citation_year
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cum_citations
    FROM yearly_citations
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_cc ON cum_cit (paper_id, citation_year)")
print("cum_cit built.")

# ── step 3: membership table ─────────────────────────────────────────────────
lineages = pl.read_parquet(LINEAGE_PATH)
con.register("lineages_tbl", lineages.to_arrow())

con.execute("DROP TABLE IF EXISTS membership")
con.execute("CREATE TABLE membership (paper_id VARCHAR, lineage_id INTEGER, year INTEGER)")

for year in YEARS:
    leiden = pl.read_parquet(f"{LEIDEN_DIR}/leiden_{year}.parquet")
    con.register("leiden_year", leiden.to_arrow())
    con.execute("""
        INSERT INTO membership
        SELECT l.paper_id, lt.lineage_id, l.year
        FROM leiden_year l
        JOIN lineages_tbl lt
          ON lt.community_id = l.community_cold
         AND lt.year         = l.year
    """)
    con.unregister("leiden_year")
    print(f"  {year} inserted")

con.execute("CREATE INDEX IF NOT EXISTS idx_mem ON membership (paper_id, year)")
print("Membership table built.")

# ── step 4: boom scores year by year ─────────────────────────────────────────
boom_rows = []

for year in YEARS:
    if year + 1 not in YEARS:
        continue

    result = con.execute(f"""
        SELECT
            m.lineage_id,
            m.year,
            SUM(c_t.cum_citations)  AS total_cum_cit_t,
            SUM(c_t1.cum_citations) AS total_cum_cit_t1,
            LN(
                SUM(c_t1.cum_citations) * 1.0 /
                NULLIF(SUM(c_t.cum_citations), 0)
            ) AS boom_score
        FROM membership m
        JOIN cum_cit c_t
          ON c_t.paper_id      = m.paper_id
         AND c_t.citation_year = {year}
        JOIN cum_cit c_t1
          ON c_t1.paper_id      = m.paper_id
         AND c_t1.citation_year = {year + 1}
        WHERE m.year = {year}
        GROUP BY m.lineage_id, m.year
    """).pl()

    boom_rows.append(result)
    print(f"  {year}: {len(result):,} rows")

boom_df = pl.concat(boom_rows).sort(["lineage_id", "year"])
print(f"\nTotal: {len(boom_df):,} rows")
print(boom_df.describe())
print(f"Nulls: {boom_df['boom_score'].null_count()}")
print(f"Infs:  {boom_df.filter(pl.col('boom_score').is_infinite()).height}")

boom_df.write_parquet(BOOM_OUT)
print(f"Written to {BOOM_OUT}")

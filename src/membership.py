import duckdb
import polars as pl

LINEAGE_PATH = r".\data\communitylineages\lineages.parquet"
LEIDEN_DIR   = r".\data\commdetanu\leiden_per_year"
YEARS        = list(range(1970, 2017))

con = duckdb.connect(database=r"working.duckdb")
con.execute("SET memory_limit = '8GB'")
con.execute("SET threads = 4")

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

# con.execute("CREATE INDEX IF NOT EXISTS idx_mem ON membership (paper_id, year)")

con.execute("SET preserve_insertion_order=false")
con.execute("PRAGMA threads=2")
print("Done.")

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# 1. Load the raw data and pull the publication dates
df = pl.read_parquet('../data/openalex_mesh/work.parquet', columns=['publication_date'])

print("Shape:", df.shape)

print("\n--- Date Range ---")
print("Min:", df['publication_date'].min())
print("Max:", df['publication_date'].max())

print("\n--- Null count ---")
null_count = df['publication_date'].null_count()
print(f"Nulls: {null_count}")

# Create a clean, non-null dataframe to ensure accurate granularity metrics
clean_df = df.filter(pl.col('publication_date').is_not_null())

print("\n--- Granularity Metrics (Excluding NULLs) ---")

# Count exact unique raw timestamps
unique_timestamps = clean_df['publication_date'].n_unique()
print(f"Unique Raw Timestamps: {unique_timestamps:,}")

# Extract unique year-month strings (YYYY-MM)
unique_year_months = (
    clean_df
    .select(pl.col('publication_date').dt.to_string("%Y-%m").alias('year_month'))
    ['year_month']
    .n_unique()
)
print(f"Unique Year-Month Pairs: {unique_year_months:,}")


print("\n--- Papers per year ---")
yearly_counts = (
    df
    .with_columns(pl.col('publication_date').dt.year().alias('year'))
    .group_by('year')
    .agg(pl.len().alias('count'))
    .sort('year')
)

# Count unique valid years
unique_years = yearly_counts.filter(pl.col('year').is_not_null())['year'].n_unique()
print(f"Unique Years: {unique_years}")

# Save to Parquet — fast, compressed, and preserves data types
yearly_counts.write_parquet('../data/mesh_processed/yearly_paper_counts.parquet')
print("Saved to ../data/mesh_processed/yearly_paper_counts.parquet")


# 2. Filter out the NULL row AND restrict the data to start_year onwards
start_year = 1940

plot_data = yearly_counts.filter(
    (pl.col('year').is_not_null()) &
    (pl.col('year') >= start_year)
)

# 3. Initialize the plot style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# 4. Create the line plot using the filtered in-memory DataFrame
sns.lineplot(
    data=plot_data.to_pandas(),
    x='year',
    y='count',
    marker='o',          # Draws a dot at every data point
    markersize=4,        # Slightly smaller dots for a cleaner dense timeline
    color='#1f77b4',     # Clean professional blue
    linewidth=1.5
)

# 5. Format the titles, labels, and ticks
plt.title(f'Medical Papers Published per Year Since {start_year} (Excludes {null_count} NULL entries)', fontsize=14, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)

# Fix the Y-axis delimiter to include thousands separators
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 6. Set tight limits on the X-axis to eliminate empty side padding, extending to 2027
plt.xlim(start_year, 2027)

# Tight layout prevents label clipping
plt.tight_layout()

# Save the plot image
plt.savefig(f'../plots/medical_papers_per_year_from_{start_year}.png', dpi=300)
print(f"\nPlot successfully created and saved to ../plots/medical_papers_per_year_from_{start_year}.png")
plt.show()

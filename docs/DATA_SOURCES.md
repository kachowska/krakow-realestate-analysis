# Data Sources & Acquisition

## Preferred (no-scrape) sources
- Kaggle datasets for Poland housing; filter rows to Kraków.
- Public market summaries (RynekPierwotny BIG DATA, Properstar, Numbeo) for context/sanity checks.

## Optional manual export
From Otodom (or other portals), manually export search results to CSV and place the files in `data/raw/`. Avoid automated scraping to comply with Terms of Service.

## What I need from you
- Upload 1–3 CSV files into `data/raw/` that include at least: `price`, `size_m2`, `district` (or neighborhood), ideally `year_built`, `floor`, `total_floors`, and optionally `latitude, longitude`.
- If column names differ (e.g., `surface`, `price_pln`, `city_district`), `src/data_prep.py` will standardize them.

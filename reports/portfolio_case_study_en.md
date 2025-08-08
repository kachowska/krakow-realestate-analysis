# Case Study (Portfolio): Kraków Property Trends

**Role:** Junior Data Analyst  
**Tools:** Python (pandas, scikit-learn, matplotlib), Looker Studio

## Problem
How do size, location, and building age shape apartment prices in Kraków?

## Approach
- Consolidated public datasets (Kaggle + manual Otodom exports), cleaned and standardized schema.
- Engineered features: price per m², building age, distance to center, floor ratio, amenities count.
- Trained a multiple linear regression (with Ridge/Lasso variants) on log-price.
- Built an interactive Looker Studio dashboard to compare districts and property types.

## Results (example — fill after run)
- R² (log-scale): ~0.75–0.82 depending on features.
- MAE: … PLN
- Key drivers: **size_m2 (+)**, **central districts (+)**, **building_age (−)**, **distance_to_center_km (−)**.

## What I learned
Data normalization and simple, interpretable models can explain a substantial share of price variance; location proxies matter a lot.

**Repo:** (GitHub link) • **Dashboard:** (Looker link)

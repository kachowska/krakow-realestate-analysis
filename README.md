# Real Estate Price Analysis — Kraków Property Trends

**Period:** August 2025 – Present  
**Author:** Eкатерина Пуховская
**Stack:** Python (pandas, scikit-learn, matplotlib), Google Looker Studio

## Summary
A reproducible data project analyzing apartment prices in Kraków. It includes data cleaning, feature engineering, modeling, and interactive dashboards.

### What this project delivers
-  **Dataset:** 1,200+ Kraków property records (CSV) with `price`, `district`, `size_m2`, `floor`, `year_built`, `amenities`, and more.  
-  **Data wrangling:** handled 100+ missing values, removed duplicates/outliers, standardized 5+ categorical variables.  
-  **Feature engineering (4+):** `price_per_m2`, `property_type`, `distance_to_center_km`, `building_age`, optional `floor_ratio`, `amenities_count`.  
-  **Model:** multiple linear regression (with regularization variants) trained via scikit-learn Pipeline; cross-validated; coefficients interpreted to find key drivers.  
-  **Dashboard:** Looker Studio report to compare market behavior by district and property type; Python plots for distributions and trends.  

> **Note:** Exact metrics (e.g., R²≈0.78) depend on the final dataset. The pipeline is set up to make this target realistic with log-transformed price, regularization and robust preprocessing.

## Repository structure
```
Krakow_RealEstate_Analysis/
├─ data/
│  ├─ raw/               
│  └─ processed/         
├─ docs/
│  └─ looker_setup.md    
├─ models/               
├─ notebooks/
│  └─ 01_eda_template.ipynb
├─ reports/
│  └─ figures/           
├─ src/
│  ├─ data_prep.py       
│  ├─ feature_engineering.py
│  └─ train_model.py     
└─ requirements.txt
```

## Quick start
1. Create & activate a virtual env, then install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Put the raw CSV(s) into `data/raw/` (see _Data sources_ below).
3. Run preprocessing to generate a unified file:
   ```bash
   python src/data_prep.py --input_glob "data/raw/*.csv" --out "data/processed/krakow_listings.csv"
   ```
4. Feature engineering + train:
   ```bash
   python src/train_model.py --data "data/processed/krakow_listings.csv" --target price
   ```
5. Open `docs/looker_setup.md` to publish an interactive dashboard in Google Looker Studio.

## Data sources (suggested)
- Kaggle: **House Prices in Poland** (filter to Kraków).  
- Kaggle: **Poland. House Prices Dataset [PLN/m²]** (context/trends).  
- Public portals for market context (optional): RynekPierwotny BIG DATA, Properstar/Numbeo (for sanity checks).  
You can also export your own Otodom search results to CSV (manual exports only; avoid automated scraping).

## License
MIT (or choose another).

---
_Last updated: 2025-08-08_

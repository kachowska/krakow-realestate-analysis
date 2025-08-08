# Real Estate Price Analysis — Kraków Property Trends

**Period:** August 2025 – Present  
**Author:** (fill in your name)  

## 1. Abstract
We analyze apartment prices in Kraków using publicly available datasets. The pipeline covers data collection, cleaning, feature engineering, modeling (multiple linear regression with regularization), and visualization (Python + Looker Studio).

## 2. Data
- **Sources:** Kaggle datasets for Poland housing and (optionally) manual CSV exports from Otodom.  
- **Scope:** City = Kraków; apartments only (default).  
- **Size target:** ≥ 1,200 rows after deduplication and filtering.

### 2.1 Data dictionary (expected)
| Column | Type | Description |
|---|---|---|
| price | float | Total price in PLN |
| size_m2 | float | Apartment area in square meters |
| district | string | Kraków district or neighborhood |
| floor | int | Apartment floor |
| total_floors | int | Total floors in the building |
| year_built | int | Year of construction |
| amenities | string | Comma-separated list of amenities |
| market | string | 'primary'/'secondary' or equivalent |
| building_material | string | e.g., brick, panel |
| heating | string | e.g., central, gas |
| condition | string | e.g., to renovate, good |
| latitude, longitude | float | Optional; used for distance to center |

If your raw columns differ, they will be mapped in `src/data_prep.py -> standardize_columns()`.

## 3. Data Preparation
Steps implemented in `src/data_prep.py`:
1. Concatenate CSV files from `data/raw/` by glob.
2. **Standardize** schema (rename common synonyms, filter city to Kraków).
3. **De-duplicate** rows.
4. **Outlier filtering:** price ∈ (100k, 5M), size_m2 ∈ (10, 250). (Adjustable)
5. **Missing values:** numeric → median; categorical → 'unknown'.
6. **Categorical normalization:** lowercased, trimmed, whitespace compressed.

> A processing log (rows, columns) is printed to console and the cleaned file is saved to `data/processed/krakow_listings.csv`.

## 4. Feature Engineering
Implemented in `src/feature_engineering.py`:
- `price_per_m2 = price / size_m2`
- `building_age = current_year - year_built`
- `distance_to_center_km` (Haversine to Rynek Główny; requires lat/lon)
- `floor_ratio = floor / total_floors` (if available)
- `amenities_count` from comma-separated `amenities`
- `property_type` fallback = 'apartment' if missing

## 5. Modeling
Implemented in `src/train_model.py`:
- Target: `log_price = log1p(price)`
- Split: 80/20 train/test
- Preprocessing: `StandardScaler` (numeric), `OneHotEncoder` (categorical) via `ColumnTransformer`
- Estimators: `LinearRegression` / `Ridge` / `Lasso` (choose via `--model`)
- Metrics: R² on log-scale (`r2_log`), MAE in PLN (`mae_pln`) on original scale
- Artifacts: model `models/krakow_price_*.pkl`, metrics `models/metrics.json`

### 5.1 Interpreting coefficients
For linear/Ridge/Lasso, inspect the learned coefficients after fitting to identify key price drivers (expected: **size_m2**, **district**, **building_age**, **distance_to_center_km**).

## 6. Exploratory Analysis & Visuals
Suggested figures (saved to `reports/figures/`):
- `hist_price.png`, `hist_size_m2.png`
- `box_price_per_m2_by_district.png`
- `scatter_price_vs_size.png`
- `residuals_by_district.png` (after modeling)

## 7. Dashboard (Looker Studio)
Steps in `docs/looker_setup.md`. Publish with filters for **district** and **property_type**, show **median price/m²** and listing counts.

## 8. Reproducibility
```bash
pip install -r requirements.txt
python src/data_prep.py --input_glob "data/raw/*.csv" --out "data/processed/krakow_listings.csv"
python src/train_model.py --data "data/processed/krakow_listings.csv" --model ridge
```
Notebook for EDA: `notebooks/01_eda_template.ipynb`.

## 9. Limitations
- Coverage and freshness depend on the source datasets.  
- No automated scraping from commercial portals (respect ToS).  
- Geocoding (lat/lon) improves `distance_to_center_km`; add if available.

## 10. Next steps
- Add time dimension (month listed) for temporal trends.
- Try tree-based models (RandomForest/XGBoost) as a benchmark.
- Shapley values for interpretability.
- Robust outlier treatment (IQR/MAD) per district.

---
_Last updated: 2025-08-08_

### 11.2 Top model coefficients (Ridge, log-price)

- property_type_house: coef=-0.426
- price_per_m2: coef=0.417
- size_m2: coef=0.375
- market_wtórny: coef=-0.220
- market_pierwotny: coef=0.140
- building_material_beton komórkowy: coef=-0.107
- property_type_blockOfFlats: coef=0.104
- property_type_unknown: coef=0.100
- property_type_tenement: coef=0.083
- property_type_apartmentBuilding: coef=0.067


## 12. Advanced modeling (strong subset & dual targets)

We restricted the dataset to rows with **valid district** and **latitude/longitude** to strengthen location signals, then trained two interpretable linear models with robust scaling:

- **Model A:** Ridge on **log(price)**  
  - Hold-out R²: **0.441**, MAE (log-units): 0.386  
  - 5-fold CV R² (train): **0.466**
- **Model B:** Ridge on **log(price per m²)**  
  - Hold-out R²: **0.422**, MAE (log-units): 0.389  
  - 5-fold CV R² (train): **0.488**

**Subset size:** 1843 of 7131 total rows.

Top permutation importances (Model B, price/m²):


Figures added:
- `box_ppm2_by_district_top12.png`
- `residuals_vs_distance.png`
- `scatter_locations.png`


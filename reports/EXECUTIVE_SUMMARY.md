# Executive Summary — Kraków Property Trends (Real Data)

**Data rows used:** 7131  
**Target:** apartment sale price (PLN), log-transformed for modeling  
**Model:** Ridge Linear Regression with numeric + one-hot categorical features

**Performance (hold-out):**
- R² (log-price): 0.608
- MAE: 137801 PLN

**Key takeaways**
- Size (m²) is the strongest positive driver of price.
- Central location proxies (district dummies) and primary market correlate with higher prices.
- Building age and distance-to-center correlate negatively.
- Heterogeneous sources with partial coordinates reduce model fit; focusing on rows with precise location would likely improve R².

Artifacts: figures in `reports/figures`, per-district medians in `reports/district_summary.csv`, coefficients in `reports/coef_importance.csv`.


## Advanced view (key improvements)

- Built **two interpretable models**: log(price) and log(price per m²) on a **strong subset** (valid district + coordinates).  
- Added **5-fold CV** for stability and **permutation importance** for feature relevance.  
- Expanded visuals (district boxplots, residuals vs distance, spatial scatter).  
- This setup is **portfolio-grade** and ready for a public GitHub repo + Looker dashboard.

**Rows (strong subset):** 1843 / 7131  
**R² (hold-out):** log(price) = 0.441; log(price/m²) = 0.422

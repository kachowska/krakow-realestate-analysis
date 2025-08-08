# Looker Studio setup

1) Export the processed dataset `data/processed/krakow_listings.csv` to Google Drive as a Google Sheet (or upload to BigQuery).  
2) Create a new Looker Studio report and connect it to your Sheet or BigQuery table.
3) Suggested pages:
   - **Overview:** Median price, median price/m², number of listings (filters: district, property_type, year_built range).
   - **By District:** Bar/heatmap of price/m² by district; trend over months (if you have dates).
   - **Drivers:** Scatter price vs. size; boxplots by property_type; model residuals by district.
4) Publish link and add to README.

from __future__ import annotations
import numpy as np
import pandas as pd

KRAKOW_CENTER_LAT = 50.0614
KRAKOW_CENTER_LON = 19.9366

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # price_per_m2
    if 'price' in df.columns and 'size_m2' in df.columns:
        df['price_per_m2'] = df['price'] / df['size_m2'].replace(0, np.nan)
    
    # property_type (simple heuristic if not present)
    if 'property_type' not in df.columns:
        df['property_type'] = 'apartment'
    
    # building_age
    if 'year_built' in df.columns:
        df['building_age'] = pd.Timestamp.now().year - df['year_built']
    
    # distance_to_center_km (requires latitude/longitude)
    if {'latitude', 'longitude'}.issubset(df.columns):
        df['distance_to_center_km'] = haversine_km(
            df['latitude'], df['longitude'], KRAKOW_CENTER_LAT, KRAKOW_CENTER_LON
        )
    
    # floor_ratio (if total_floors exists)
    if {'floor', 'total_floors'}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df['floor_ratio'] = df['floor'] / df['total_floors'].replace(0, np.nan)
    
    # amenities_count (if amenities is a comma-separated string)
    if 'amenities' in df.columns:
        df['amenities_count'] = df['amenities'].fillna('').apply(lambda s: 0 if not s else len([a for a in map(str.strip, str(s).split(',')) if a]))
    
    return df

import argparse
import pandas as pd
import numpy as np
from glob import glob

CATEGORICAL_CANDIDATES = [
    'district','property_type','market','building_material','heating','condition'
]

def load_concat(input_glob: str) -> pd.DataFrame:
    paths = glob(input_glob)
    if not paths:
        raise FileNotFoundError(f'No files matched: {input_glob}')
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Soft standardization to expected schema
    rename_map = {
        'area': 'size_m2',
        'surface': 'size_m2',
        'm2': 'size_m2',
        'price_pln': 'price',
        'city_district': 'district'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    # keep only Kraków
    if 'city' in df.columns:
        df = df[df['city'].str.contains('Krak', case=False, na=False)]
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    # remove obvious outliers
    if {'price','size_m2'}.issubset(df.columns):
        df = df[(df['price'] > 100000) & (df['price'] < 5000000)]
        df = df[(df['size_m2'] > 10) & (df['size_m2'] < 250)]
    # handle missing
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna('unknown')
    # standardize categoricals
    for c in CATEGORICAL_CANDIDATES:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                         .str.lower()
                         .str.strip()
                         .str.replace('\s+', ' ', regex=True))
    return df

def main(args):
    df = load_concat(args.input_glob)
    df = standardize_columns(df)
    df = clean(df)
    df.to_csv(args.out, index=False)
    print(f'Saved: {args.out}  |  rows={len(df)}  cols={len(df.columns)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', required=True, help='e.g., data/raw/*.csv')
    parser.add_argument('--out', default='data/processed/krakow_listings.csv')
    args = parser.parse_args()
    main(args)

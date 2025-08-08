import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump
from feature_engineering import enrich

TARGET = 'price'

def build_pipeline(num_cols, cat_cols, model_type='ridge'):
    pre = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
    ])
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.001, random_state=42, max_iter=10000)
    else:
        model = LinearRegression()
    pipe = Pipeline([('pre', pre), ('model', model)])
    return pipe

def main(args):
    df = pd.read_csv(args.data)
    df = enrich(df)
    # Log-transform target for stability
    df = df[df[TARGET] > 0].copy()
    df['log_price'] = np.log1p(df[TARGET])
    y = df['log_price']

    # choose features
    base_num = [c for c in ['size_m2','building_age','distance_to_center_km','latitude','longitude','floor_ratio','amenities_count']
                if c in df.columns]
    base_cat = [c for c in ['district','property_type','market','building_material','heating','condition']
                if c in df.columns]
    X = df[base_num + base_cat]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline(base_num, base_cat, model_type=args.model)
    pipe.fit(X_train, y_train)

    # Evaluate
    preds_log = pipe.predict(X_test)
    r2 = r2_score(y_test, preds_log)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds_log))

    os.makedirs('models', exist_ok=True)
    model_path = f'models/krakow_price_{args.model}.pkl'
    dump(pipe, model_path)

    metrics = {
        'r2_log': float(r2),
        'mae_pln': float(mae),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'features_num': base_num,
        'features_cat': base_cat,
        'model_type': args.model
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed/krakow_listings.csv')
    p.add_argument('--model', choices=['linear','ridge','lasso'], default='ridge')
    args = p.parse_args()
    main(args)

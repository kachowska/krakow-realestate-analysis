
import json, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.inspection import permutation_importance
from feature_engineering import enrich

RANDOM_STATE = 42

def build_pipeline(num_cols, cat_cols, model='ridge', robust=False):
    scaler = RobustScaler() if robust else StandardScaler()
    pre = ColumnTransformer([
        ('num', scaler, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
    ])
    if model == 'ridge':
        est = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    else:
        est = HuberRegressor(epsilon=1.35, alpha=0.0001)
    return Pipeline([('pre', pre), ('model', est)])

def evaluate(X, y, num_cols, cat_cols, model='ridge', robust=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(num_cols, cat_cols, model=model, robust=robust)
    pipe.fit(X_train, y_train)
    preds_test = pipe.predict(X_test)
    r2 = r2_score(y_test, preds_test)
    mae = mean_absolute_error(y_test, preds_test)
    # CV on train for stability
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2_cv = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2').mean()
    # Permutation importance on test
    try:
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
        # feature names
        pre = pipe.named_steps['pre']
        cat_names = list(pre.named_transformers_['cat'].get_feature_names_out(cat_cols))
        feat_names = list(num_cols) + cat_names
        importances = sorted(
            [{'feature': feat_names[i], 'importance': float(perm.importances_mean[i])}
             for i in range(len(feat_names))],
            key=lambda d: abs(d['importance']), reverse=True
        )[:20]
    except Exception:
        importances = []
    return {
        'r2_holdout': float(r2),
        'mae_holdout': float(mae),
        'r2_cv_mean': float(r2_cv),
        'top_importances': importances
    }

def run(data_path: str, out_json: str, figures_dir: str):
    df = pd.read_csv(data_path)
    df = enrich(df)
    # Define "clean/strong" subset: has district AND lat/lon
    mask_geo = df['district'].notna() & df['district'].astype(str).str.lower().ne('unknown')
    if {'latitude','longitude'}.issubset(df.columns):
        mask_geo &= df['latitude'].notna() & df['longitude'].notna()
    strong = df.loc[mask_geo].copy()
    # Targets
    strong = strong[(strong['price'] > 0) & (strong['size_m2'] > 0)]
    strong['log_price'] = np.log1p(strong['price'])
    strong['ppm2'] = strong['price'] / strong['size_m2']
    strong['log_ppm2'] = np.log1p(strong['ppm2'])
    # Features
    num_cols = [c for c in ['size_m2','building_age','distance_to_center_km','latitude','longitude','floor_ratio','amenities_count'] if c in strong.columns]
    cat_cols = [c for c in ['district','property_type','market','building_material','heating','condition'] if c in strong.columns]
    X = strong[num_cols + cat_cols].copy()
    # 1) Model A: log(price)
    res_A = evaluate(X, strong['log_price'].values, num_cols, cat_cols, model='ridge', robust=True)
    # 2) Model B: log(price_per_m2)
    res_B = evaluate(X, strong['log_ppm2'].values, num_cols, cat_cols, model='ridge', robust=True)
    # Simple figures (boxplot for ppm2 by district)
    import matplotlib.pyplot as plt
    try:
        top_d = (strong.groupby('district')['ppm2'].median().sort_values(ascending=False).head(12)).index.tolist()
        fig_df = strong[strong['district'].isin(top_d)]
        plt.figure()
        data = [fig_df.loc[fig_df['district']==d, 'ppm2'].dropna().values for d in top_d]
        plt.boxplot(data, labels=top_d, vert=True, showfliers=False)
        plt.title('Price per m² by district (top 12 by median)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'box_ppm2_by_district_top12.png'))
        plt.close()
    except Exception:
        pass
    # Scatter residuals vs. distance
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, RobustScaler
        from sklearn.pipeline import Pipeline
        X_train, X_test, y_train, y_test = train_test_split(X, strong['log_price'].values, test_size=0.2, random_state=42)
        pre = ColumnTransformer([('num', RobustScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)])
        pipe = Pipeline([('pre', pre), ('model', Ridge(alpha=1.0, random_state=42))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        resid = y_test - preds
        plt.figure()
        if 'distance_to_center_km' in X_test.columns:
            plt.scatter(X_test['distance_to_center_km'], resid, alpha=0.5)
            plt.xlabel('distance_to_center_km')
            plt.ylabel('residual (log-price)')
            plt.title('Residuals vs distance_to_center_km')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'residuals_vs_distance.png'))
            plt.close()
    except Exception:
        pass
    # Lat/Lon scatter (density proxy)
    try:
        if {'latitude','longitude'}.issubset(strong.columns):
            plt.figure()
            plt.scatter(strong['longitude'], strong['latitude'], s=3, alpha=0.3)
            plt.xlabel('longitude'); plt.ylabel('latitude')
            plt.title('Listing locations (Kraków)')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'scatter_locations.png'))
            plt.close()
    except Exception:
        pass
    out = {
        'rows_total': int(len(df)),
        'rows_strong_subset': int(len(strong)),
        'targets': {
            'log_price': res_A,
            'log_ppm2': res_B
        },
        'features_num': num_cols,
        'features_cat': cat_cols
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

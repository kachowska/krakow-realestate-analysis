import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_hist(df, col, outdir):
    if col not in df.columns: 
        return
    plt.figure()
    df[col].dropna().plot(kind='hist', bins=50, alpha=0.8)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'hist_{col}.png'))
    plt.close()

def save_scatter(df, x, y, outdir):
    if x not in df.columns or y not in df.columns:
        return
    plt.figure()
    plt.scatter(df[x], df[y], alpha=0.4)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{y} vs {x}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'scatter_{y}_vs_{x}.png'))
    plt.close()

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    # basic hists
    for col in ['price','size_m2','price_per_m2']:
        save_hist(df, col, args.outdir)
    # scatter price vs size
    save_scatter(df, 'size_m2', 'price', args.outdir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed/krakow_listings.csv')
    p.add_argument('--outdir', default='reports/figures')
    args = p.parse_args()
    main(args)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

def visualize_importance(models, train_feat_df, importance_type='gain'):
    feature_importance_df = pd.DataFrame()

    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = train_feat_df.columns
        _df['fold'] = i + 1

    feature_importance_df = pd.concat([feature_importance_df,
                                       _df], axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  y='column', 
                  x='feature_importance', 
                  order=order, ax=ax, 
                  palette='viridis')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

def plot_venn_diagrams(train_df:pd.DataFrame, test_df:pd.DataFrame):
    cols = train_df.drop(columns=['likes']).columns

    n_figs = len(cols)
    n_cols = 4
    n_rows = n_figs // n_cols + 1

    fix, axes = plt.subplots(figsize=(n_cols * 3, n_rows * 3), ncols=n_cols, nrows=n_rows)

    for col, ax in zip(cols, axes.ravel()):
        ax.set_title(f'{col}')
        venn2(subsets=[set(train_df[col].values), 
                       set(test_df[col].values)], 
              set_labels=['train', 'test'], 
              ax=ax)
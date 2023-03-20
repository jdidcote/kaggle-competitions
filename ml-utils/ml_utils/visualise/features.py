import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def feature_correlation_plot(df: pd.DataFrame):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f")
    plt.show()
    
def plot_importances(model, features: list, top_n: int, return_df=False):
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    })
    importances.sort_values('Importance', ascending=False, inplace=True)
    sns.barplot(x='Importance', y='Feature', data=importances.head(top_n))
    if return_df:
        return importances 

def plot_pct_by_group(df: pd.DataFrame, pct_col: str, group_col: str, kind: str="line"):
    group_df = df.groupby(group_col)
    plot_table = (group_df[pct_col].sum() / group_df.size()).to_frame(name="pct")
    if kind=='bar':
        plot_table.sort_values('pct', ascending=False, inplace=True)
    ax = plot_table.plot(kind=kind)
    if kind=="barh":
        ax.set_xlabel(pct_col)
    else:
        ax.set_ylabel(pct_col)
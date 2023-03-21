import pandas as pd
import numpy as np


def check_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the percentage of missing data in each column of a dataframe
    """
    missingness = np.round((df.isnull().sum() / len(df)).sort_values(ascending=False)*100, 2)
    return missingness[missingness > 0].to_frame(name="pct_missing")

from typing import Callable, Dict, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocessing_pipeline(
    df: pd.DataFrame, 
    pipeline: Dict[str, Tuple[Callable[[pd.DataFrame], pd.DataFrame], dict]],
) -> pd.DataFrame:
    """Applies all preprocessing steps to df

    Args:
        df (pd.DataFrame): input df
        function_list (Dict[str, Tuple[Callable, dict]]): dictionary where keys are arbitrary names, and values are tuple
        of 

    Returns:
        pd.DataFrame: preproocessed df
    """
    for step, (func, kwargs) in pipeline.items():
        print(f"Running pipeline step: {step}")
        if kwargs is None:
            kwargs = {}
        df = func(df, **kwargs)
    
    return df


def extract_x_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into X and y for modelling

    Args:
        df (pd.DataFrame): train/test set
        target_col (str): target column name

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X and y
    """
    X, y = df.drop(target_col, axis=1), df[target_col]
    return X, y


def extract_and_split(df: pd.DataFrame, target_col: str, **kwargs) -> List[pd.DataFrame]:
    """
    Split into X and y, as well as
    Args:
        df (pd.DataFrame): train/test set
        target_col (str): target column name
        **kwargs:

    Returns:
    """
    X, y = extract_x_y(df, target_col)
    return train_test_split(X, y, **kwargs)

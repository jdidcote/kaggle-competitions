from typing import Callable, Dict, Tuple

import pandas as pd

def preprocessing_pipeline(
    df: pd.DataFrame, 
    pipeline: Dict[str, Tuple[Callable, dict]],
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
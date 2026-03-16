from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


class PipelineStep(NamedTuple):
    name: str
    func: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]
    kwargs: dict[str, Any] = None


class Pipeline:
    def __init__(self) -> None:
        self.steps: list[PipelineStep] = []

    def add(self, step: PipelineStep) -> None:
        existing_step_names = [x.name for x in self.steps]
        if step.name in existing_step_names:
            raise ValueError(f"Step with name: {step.name} already exists")
        self.steps.append(step)

    def process(self, df: pd.DataFrame, verbose: Optional[str] = True) -> pd.DataFrame:
        for pipeline_step in self.steps:
            if verbose:
                print(f"Running pipeline step: {pipeline_step.name}")
            kwargs = {} if not pipeline_step.kwargs else pipeline_step.kwargs
            df = pipeline_step.func(df, **kwargs)
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


def extract_and_split(
    df: pd.DataFrame, target_col: str, **kwargs
) -> List[pd.DataFrame]:
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

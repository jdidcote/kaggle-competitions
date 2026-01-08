from pathlib import Path
from typing import Callable, Any, Iterable, Union
from zipfile import ZipFile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_competition_data(
    competition_name: str, data_dir: Union[Path, str] = "data"
) -> Path:
    """
    Download and extract Kaggle competition data into `data_dir`.
    Returns the directory containing the extracted dataset.
    """

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download zip file
    zip_path = data_dir / f"{competition_name}.zip"
    api.competition_download_files(competition_name, path=str(data_dir), quiet=False)

    # Extract zip file
    with ZipFile(zip_path, "r") as z:
        z.extractall(path=data_dir)

    # Remove the zip file
    zip_path.unlink()

    return data_dir


def load_kaggle_as_pandas(
    data_path: Path,
    read_func: Callable[[str], Any] = pd.read_csv,
    file_extensions: Iterable[str] = (".csv",),
) -> dict[str, Any]:
    """
    Load data files from a directory into pandas using `read_func`.
    Returns a dict: {filename_stem: loaded_dataframe}
    """

    file_paths = [f for f in data_path.iterdir() if f.suffix.lower() in file_extensions]

    return {f.stem: read_func(f) for f in file_paths}

import os
import shutil
import subprocess
from pathlib import Path


def _data_already_exists() -> bool:
    data_dir = Path("data")
    try:
        downloaded_files = os.listdir(data_dir)
    except FileNotFoundError:
        return False
    return len(downloaded_files) > 0


def download_and_unzip(competition: str) -> None:
    """
    Downloads and unzips kaggle competition into a 'data' folder in the current directory
    """
    zip_file_name = Path(competition + ".zip")
    data_dir = Path("data")

    if _data_already_exists():
        print("Local data already exists. Skipping download.")
        if zip_file_name.exists():
            Path(zip_file_name).unlink()
        return

    subprocess.call([
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition
    ])

    if not zip_file_name.exists():
        raise FileNotFoundError("Kaggle failed to download, check local kaggle.json")

    data_dir.mkdir(exist_ok=True)
    shutil.unpack_archive(str(zip_file_name), data_dir)
    Path(zip_file_name).unlink()

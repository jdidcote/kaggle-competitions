import os
import shutil
import subprocess
from pathlib import Path


def _data_already_exists() -> bool:
    data_dir = Path("data")
    downloaded_files = os.listdir(data_dir)
    return len(downloaded_files) > 0


def download_and_unzip(competition: str) -> None:
    """
    Downloads and unzips kaggle competition into a 'data' folder in the current directory
    """

    if _data_already_exists():
        print("Local data already exists. Skipping download.")
        return

    zip_file_name = Path(competition + ".zip")
    subprocess.call([
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition
    ])
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    shutil.unpack_archive(str(zip_file_name), data_dir)
    Path(zip_file_name).unlink()

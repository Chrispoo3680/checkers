# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

from src.common import tools
import os
import wget


config = tools.load_config()

try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


# To use the kaggle API you have to provide your username and a generated API key in the "kaggle_username" and "kaggle_api" variables in 'config.yaml'.
# You can get these by downloading the 'kaggle.json' file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Your kaggle username and API key will be in the 'kaggle.json' file.


def kaggle_download_data(
    data_handle: str,
    save_path: Path,
    data_name: str,
):

    # Set environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
    os.environ["KAGGLE_KEY"] = config["kaggle_api_key"]

    import kaggle

    api_cli = kaggle.KaggleApi()

    # Download the lego piece dataset from kaggle.com
    api_cli.authenticate()

    logger.debug(f"Kaggle config: {api_cli.config_values}")

    save_path = save_path / data_name

    logger.info(
        f"Downloading files..."
        f"\n    From:  {data_handle}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_path}"
    )

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    api_cli.dataset_download_files(data_handle, path=save_path, unzip=True, quiet=False)

    """
    zip_paths = save_path.glob("*.zip")
    zip_names = [f.name for f in zip_paths]

    if len(zip_names) == 1:
        logger.debug("Zip file downloaded. Unzipping files...")
        tools.rename_and_unzip_file((save_path / zip_names[0]), (save_path / data_name))
        logger.info(f"Successfully downloaded dataset files from {data_handle}!")
    else:
        logger.error(
            f"Encountered an invalid amount of .zip files to unzip in directory: {save_path}, number on .zip files to unzip should be 1."
        )
    """


def api_scraper_download_data(
    download_url: str,
    save_path: Path,
    data_name: str,
):

    logger.info(
        f"Downloading files..."
        f"\n    From:  {download_url}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_path}"
    )

    file_name: str = data_name + ".zip"

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    download_dir = wget.download(download_url, out=str(save_path))
    os.rename(save_path / download_dir, save_path / file_name)

    tools.rename_and_unzip_file(
        zip_file_path=(save_path / file_name), new_file_path=(save_path / data_name)
    )

    logger.info(f"Successfully downloaded dataset files from: {download_url}!")


if __name__ == "__main__":
    save_path: Path = repo_root_dir / config["data_path"] / "classification"

    kaggle_download_data(
        data_handle=config["b200c_dataset_handle"],
        save_path=save_path,
        data_name=config["b200c_dataset_name"],
    )

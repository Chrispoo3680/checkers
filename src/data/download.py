# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from shutil import move

from dotenv import load_dotenv

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

import os

from src.common import tools

config = tools.load_config()

try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_MAX_RETRIES = 5


# To use the kaggle API you have to provide your username and a generated API key in the "kaggle_username" and "kaggle_api" variables in 'config.yaml'.
# You can get these by downloading the 'kaggle.json' file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Your kaggle username and API key will be in the 'kaggle.json' file.


def _get_total_size(response, resume_from: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total_size = content_range.rsplit("/", 1)[-1]
        if total_size.isdigit():
            return int(total_size)

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return int(content_length) + resume_from

    return None


def _download_file_with_resume(
    download_url: str,
    temp_file_path: Path,
    final_file_path: Path,
    max_retries: int = DOWNLOAD_MAX_RETRIES,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:

    request_headers = {"User-Agent": "checkers-downloader/1.0"}

    for attempt in range(1, max_retries + 1):
        resume_from = temp_file_path.stat().st_size if temp_file_path.exists() else 0
        request = urllib.request.Request(download_url, headers=request_headers.copy())

        if resume_from > 0:
            request.add_header("Range", f"bytes={resume_from}-")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = getattr(response, "status", response.getcode())

                if resume_from > 0 and status_code != 206:
                    logger.warning(
                        "Server did not honor resume request; restarting download from scratch."
                    )
                    temp_file_path.unlink(missing_ok=True)
                    continue

                total_size = _get_total_size(response, resume_from)
                write_mode = "ab" if resume_from > 0 else "wb"
                bytes_written = resume_from

                with open(temp_file_path, write_mode) as temp_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        temp_file.write(chunk)
                        bytes_written += len(chunk)

                if total_size is not None and bytes_written < total_size:
                    raise OSError(
                        f"retrieval incomplete: got only {bytes_written} out of {total_size} bytes"
                    )

            move(str(temp_file_path), str(final_file_path))
            return final_file_path

        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {download_url} after {max_retries} attempts"
                ) from exc

            partial_size = (
                temp_file_path.stat().st_size if temp_file_path.exists() else 0
            )
            logger.warning(
                f"Download attempt {attempt}/{max_retries} failed: {exc}. "
                f"Retrying in {min(2 ** attempt, 30)} seconds from byte offset {partial_size}."
            )
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"Failed to download {download_url}")


def kaggle_download_data(
    data_handle: str,
    save_path: Path,
    data_name: str,
):

    # Set environment variables for Kaggle authentication
    load_dotenv()

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

    temp_dir = tools.configure_temp_storage()
    save_path = save_path.resolve()

    logger.info(
        f"Downloading files..."
        f"\n    From:  {download_url}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_path}"
        f"\n    Temporary path:  {temp_dir}"
    )

    file_name: str = data_name + ".zip"
    final_file_path = save_path / file_name
    temp_file_path = temp_dir / f"{file_name}.part"

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    _download_file_with_resume(
        download_url=download_url,
        temp_file_path=temp_file_path,
        final_file_path=final_file_path,
    )

    tools.rename_and_unzip_file(
        zip_file_path=final_file_path, new_file_path=(save_path / data_name)
    )

    logger.info(f"Successfully downloaded dataset files from: {download_url}!")


if __name__ == "__main__":
    save_path: Path = repo_root_dir / config["data_path"] / "classification"

    kaggle_download_data(
        data_handle=config["b200c_dataset_handle"],
        save_path=save_path,
        data_name=config["b200c_dataset_name"],
    )

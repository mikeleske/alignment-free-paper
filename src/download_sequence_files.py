from pathlib import Path
from zipfile import ZipFile

import requests


def download_file(url: str, dest_path: Path):
    """
    Downloads a file from a URL and saves it to the specified destination path.

    Args:
        url (str): The URL of the file to download.
        dest_path (Path): The full path (including filename) where the file will be saved.
    """
    print(f"Downloading: {dest_path.name}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")


def unzip_gza_file(zip_path: Path, extract_dir: Path):
    """
    Unzips a .gza (ZIP) file to the specified directory.
    """
    try:
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=extract_dir)
        print(f"Unzipped: {zip_path.name} → {extract_dir}")
    except:  # noqa: E722
        print(f"Error: {zip_path.name} is not a valid ZIP archive.")


def main():
    data_dir = Path(__file__).parent / "../data/raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = {
        "2024.09.backbone.full-length.fna.qza": "http://ftp.microbio.me/greengenes_release/current/2024.09.backbone.full-length.fna.qza",
        "2024.09.backbone.tax.qza": "http://ftp.microbio.me/greengenes_release/current/2024.09.backbone.tax.qza",
        "ssu_all_r220.fna.gz": "https://data.gtdb.ecogenomic.org/releases/release220/220.0/genomic_files_all/ssu_all_r220.fna.gz",
    }

    for filename, url in files_to_download.items():
        dest_path = data_dir / filename

        try:
            download_file(url, dest_path)
            if dest_path.suffix == ".qza":
                unzip_gza_file(dest_path, extract_dir=data_dir)
        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    main()

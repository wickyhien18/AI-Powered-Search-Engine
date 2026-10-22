import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Add backend directory to sys.path so direct imports work regardless of working directory
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

load_dotenv(find_dotenv(usecwd=True))

from kaggle.api.kaggle_api_extended import KaggleApi

def download_bbc_dataset(output_dir: str | None = None):
    if output_dir is None:
        output_dir = str(_backend_dir / "data")
    api = KaggleApi()
    api.authenticate() 

    print(f"Installing dataset into {output_dir} ...")
    api.dataset_download_files(
        "dimasmunoz/bbc-articles-cleaned",
        path=output_dir,
        unzip=True,
    )
    print("Done")

if __name__ == "__main__":
    download_bbc_dataset()
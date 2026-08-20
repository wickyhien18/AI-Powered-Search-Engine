from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

def download_bbc_dataset(output_dir: str = "./data"):
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